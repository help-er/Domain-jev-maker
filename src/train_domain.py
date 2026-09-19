"""Train a domain decision model: LoRA + pointer head, cross-entropy to soft targets.

Layout: every option is an isolated branch over a shared `[state + question]`
prefix, all branches at identical positions, so option representations are
exchangeable by construction and option order cannot affect the answer.

Objective: cross-entropy to the record's soft target. For a single-step
decision this is the proper scoring rule you want -- it is uniquely minimised
at p = q -- and no RL machinery is required to reach it.

Usage:
  python src/train_domain.py --data-dir data/banking --adapter models/banking
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import BASE, SEED
from pointer import PointerHead, logits_for
from sym_options import pack_options


class DomainDataset(Dataset):
    """Records are self-describing: each carries its own `criteria` map, so a
    new domain needs no code change.

    `max_options` subsamples distractors during TRAINING only. Long option
    lists are the dominant cost -- an explicit 4D mask forces the attention
    implementation onto a path that materialises a T x T matrix per head per
    layer, so step time grows sharply with K. Keeping every option that carries
    target mass plus a random sample of the rest cuts it several-fold, and is
    sound because the pointer scores each option independently from its own
    representation. Inference always uses the full set.
    """

    def __init__(self, path, tok, limit=None, report=True, max_options=None,
                 seed=SEED, hard_targets=False):
        self.items = []
        rng = random.Random(seed)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            recs = recs[:limit]
        n_sub = 0
        for r in recs:
            disp2canon = {v: k for k, v in r["vocab_map"].items()}
            canon_order = [disp2canon[o] for o in r["options"]]
            crit = dict(r["criteria"])
            options = [crit[c] for c in canon_order]          # label-free
            if max_options and len(canon_order) > max_options:
                keep = [i for i, c in enumerate(canon_order)
                        if r["soft"].get(c, 0.0) > 0]         # never drop mass
                pool = [i for i in range(len(canon_order)) if i not in keep]
                rng.shuffle(pool)
                keep = sorted(keep + pool[:max(0, max_options - len(keep))])
                canon_order = [canon_order[i] for i in keep]
                options = [options[i] for i in keep]
                n_sub += 1
            ids, pos, mask, opt_last, dec = pack_options(
                tok, r["text"], r["question"], options)
            tgt = [r["soft"][c] for c in canon_order]
            if hard_targets:
                # ablation arm: collapse every soft target onto its argmax
                m = max(range(len(tgt)), key=lambda j: tgt[j])
                tgt = [1.0 if j == m else 0.0 for j in range(len(tgt))]
            t = sum(tgt)
            self.items.append({
                "ids": ids.tolist(), "pos": pos.tolist(), "mask": mask,
                "opt": opt_last, "dec": dec,
                "target": [x / t for x in tgt],
                "canon": list(canon_order),
                "kind": r.get("kind", r.get("task", "item")),
            })
        if report:
            H = [-sum(p * math.log(p) for p in i["target"] if p > 0)
                 for i in self.items]
            L = [len(i["ids"]) for i in self.items]
            print(f"  {path}: n={len(self.items)} "
                  f"(option-subsampled {n_sub}) "
                  f"seq min/med/max={min(L)}/{sorted(L)[len(L)//2]}/{max(L)} "
                  f"mean target entropy={sum(H)/len(H):.3f}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    T = max(len(b["ids"]) for b in batch)
    K = max(len(b["opt"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    m = torch.zeros((B, 1, T, T), dtype=torch.bool)
    dec = torch.zeros(B, dtype=torch.long)
    opt = torch.zeros((B, K), dtype=torch.long)
    tgt = torch.zeros((B, K), dtype=torch.float)
    valid = torch.zeros((B, K), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        pos[i, :n] = torch.tensor(b["pos"])
        m[i, 0, :n, :n] = b["mask"]
        # a fully-masked row makes softmax produce NaN, so let each padding
        # token attend to itself; its output is discarded anyway
        for t in range(n, T):
            m[i, 0, t, t] = True
        dec[i] = b["dec"]
        k = len(b["opt"])
        opt[i, :k] = torch.tensor(b["opt"])
        tgt[i, :k] = torch.tensor(b["target"])
        valid[i, :k] = True
    return {"input_ids": ids, "position_ids": pos, "mask4d": m,
            "dec": dec, "opt": opt, "target": tgt, "valid": valid}


def soft_loss(z, batch, device):
    valid = batch["valid"].to(device)
    logp = F.log_softmax(z, -1)
    q = batch["target"].to(device).float().masked_fill(~valid, 0.0)
    return -(q * logp.masked_fill(~valid, 0.0)).sum(-1).mean()


@torch.no_grad()
def evaluate(model, head, ds, device, dtype, pad_id, bs=4):
    model.eval()
    head.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False,
                    collate_fn=lambda b: collate(b, pad_id))
    tot_kl = tot_ce = acc = n = 0
    xs, ys = [], []
    for batch in dl:
        z = logits_for(model, head, batch, device, dtype)
        valid = batch["valid"].to(device)
        logp = F.log_softmax(z, -1)
        p = logp.exp()
        q = batch["target"].to(device).float().masked_fill(~valid, 0.0)
        ce = -(q * logp.masked_fill(~valid, 0.0)).sum(-1)
        hq = -(q * torch.log(q.clamp_min(1e-12))).sum(-1)
        hp = -(p * logp.masked_fill(~valid, 0.0)).sum(-1)
        tot_ce += ce.sum().item()
        tot_kl += (ce - hq).sum().item()
        acc += (p.argmax(-1) == q.argmax(-1)).sum().item()
        xs += hq.tolist()
        ys += hp.tolist()
        n += q.size(0)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    model.train()
    head.train()
    return {"n": n, "kl": tot_kl / n, "ce": tot_ce / n, "acc": acc / n,
            "h_pred": my, "h_true": mx,
            "r": num / (dx * dy) if dx and dy else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="uses <dir>/train.jsonl and <dir>/heldout.jsonl")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--probe", action="store_true",
                    help="3 steps on 48 items, to measure step time")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-options", type=int, default=10,
                    help="subsample distractors in TRAINING only; 0 disables")
    ap.add_argument("--hard-targets", action="store_true",
                    help="ablation: collapse soft targets to one-hot")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("datasets:", flush=True)
    tr = DomainDataset(os.path.join(a.data_dir, "train.jsonl"), tok,
                       limit=48 if a.probe else None,
                       max_options=a.max_options or None,
                       hard_targets=a.hard_targets)
    he = DomainDataset(os.path.join(a.data_dir, "heldout.jsonl"), tok)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                     lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg).to(device)
    head = PointerHead(model.config.hidden_size).to(device).float()

    dl = DataLoader(tr, batch_size=a.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = max(1, (len(dl) // a.accum) * a.epochs)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": a.lr},
        {"params": head.parameters(), "lr": a.head_lr}])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.head_lr], total_steps=steps, pct_start=0.05)

    print(f"\ntraining: {steps} steps", flush=True)
    t0, step = time.time(), 0
    for _ in range(a.epochs):
        for i, batch in enumerate(dl):
            z = logits_for(model, head, batch, device, dtype)
            (soft_loss(z, batch, device) / a.accum).backward()
            if (i + 1) % a.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"  step {step}/{steps} "
                          f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
                if a.probe and step >= 3:
                    print(f"PROBE: {(time.time()-t0)/step:.1f}s/step")
                    return
    print(f"\nTRAIN DONE: {step} steps in {time.time()-t0:.0f}s", flush=True)

    st = evaluate(model, head, he, device, dtype, tok.pad_token_id)
    print("\n" + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in st.items()))
    os.makedirs(a.adapter, exist_ok=True)
    model.save_pretrained(a.adapter)
    torch.save(head.state_dict(), os.path.join(a.adapter, "pointer_head.pt"))
    json.dump(st, open(os.path.join(a.adapter, "eval.json"), "w"), indent=2)
    print(f"saved to {a.adapter}", flush=True)


if __name__ == "__main__":
    main()
