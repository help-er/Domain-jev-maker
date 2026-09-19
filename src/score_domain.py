"""Score a trained adapter on a held-out split, naming real labels.

Emits one JSON row per item with the canonical label for every option, the
item kind, and the full distribution, so the output can be compared class by
class against another system and broken down by kind.

Usage:
  python src/score_domain.py --adapter models/banking --data-dir data/banking \
      --out out/banking-local.jsonl
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_domain as T
from config import BASE
from pointer import PointerHead, logits_for

EPS = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if a.dtype == "fp32" or device == "cpu" \
        else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, a.adapter).to(device).eval()

    head = PointerHead(model.config.hidden_size).to(device).float()
    head.load_state_dict(torch.load(os.path.join(a.adapter, "pointer_head.pt")))
    head.eval()

    # max_options=None: inference always uses the full option set
    ds = T.DomainDataset(os.path.join(a.data_dir, a.split + ".jsonl"), tok,
                         max_options=None)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False,
                    collate_fn=lambda b: T.collate(b, tok.pad_token_id))

    rows, idx = [], 0
    with torch.no_grad():
        for batch in dl:
            z = logits_for(model, head, batch, device, dtype)
            valid = batch["valid"].to(device)
            p = F.log_softmax(z, -1).exp()
            for bi in range(z.size(0)):
                it = ds.items[idx]
                k = int(valid[bi].sum())
                canon = it["canon"]
                pv = p[bi, :k].tolist()
                qv = list(it["target"])
                s = sum(qv) or 1.0
                qv = [x / s for x in qv]
                ce = -sum(q * math.log(max(pp, EPS)) for q, pp in zip(qv, pv))
                hq = -sum(x * math.log(max(x, EPS)) for x in qv if x > 0)
                hp = -sum(x * math.log(max(x, EPS)) for x in pv if x > 0)
                am = max(range(k), key=lambda j: pv[j])
                aq = max(range(k), key=lambda j: qv[j])
                rows.append({"i": idx, "task": it["kind"], "k": k,
                             "correct": am == aq, "conf": max(pv),
                             "pred": canon[am], "gold": canon[aq],
                             "probs": {canon[j]: pv[j] for j in range(k)},
                             "true": {canon[j]: qv[j] for j in range(k)},
                             "kl": ce - hq, "ce": ce,
                             "h_true": hq, "h_pred": hp})
                idx += 1
            if idx % 200 < a.bs:
                print(f"  {idx}/{len(ds)}", flush=True)

    out = a.out or os.path.join("out", os.path.basename(a.adapter) + ".jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    print(f"\n{n} rows -> {out}")
    print(f"  acc {sum(r['correct'] for r in rows)/n:.4f}   "
          f"KL {sum(r['kl'] for r in rows)/n:.4f}   "
          f"H_pred {sum(r['h_pred'] for r in rows)/n:.4f}")


if __name__ == "__main__":
    main()
