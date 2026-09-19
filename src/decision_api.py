"""The decision request/response contract, served locally.

One forward pass answers Q typed questions about a shared state, where:

  * the state is encoded once
  * questions cannot read each other
  * options within a question are exchangeable
  * probabilities come from a pointer readout

The packed sequence is a three-level tree:

    [ state ][ instr_1 ][ o_1,1 ][ o_1,2 ]...[ tail_1 ][ instr_2 ][ o_2,1 ]...[ tail_2 ]

    state token       -> keys 0..t                           (causal)
    instr_i token     -> state + own instruction, causally
    option (i,j)      -> state + instr_i + own span           (no sibling option,
                                                               no other question)
    tail_i token      -> state + instr_i + ALL options of i + own tail

Positions: state 0..S-1; instr_i S..S+Qi-1; EVERY option of question i starts at
S+Qi; tail_i at S+Qi+max_j|o_ij|. So options are exchangeable and questions are
isolated, both by construction rather than by training.

The contract:

  * typed questions: `choice` (with per-option `criteria`), `noul`, `score`
  * question identifiers are NOT sent to the model
  * choice confidence is arithmetic on the distribution, not a learned estimate:
        c = (p_max - 1/K) / (1 - 1/K)      for K > 1, and 1 for K = 1
  * per-branch limit ~32,768 tokens; per-request ~65,536, counting the state once
  * `output_tokens` counts the serialised response, not neural decoding steps

Usage:
  python src/decision_api.py --demo
  python src/decision_api.py --selftest
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from config import BASE, SYSTEM
from pointer import PointerHead

BRANCH_LIMIT = 32768
REQUEST_LIMIT = 65536

STATE_TMPL = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
              "<|im_start|>user\n"
              "State:\n{state}\n\n")
INSTR_TMPL = "Question: {instructions}\nAllowed options:\n"
OPT_TMPL = "- {text}\n"
TAIL = "Decision:<|im_end|>\n<|im_start|>assistant\n"


# ------------------------------------------------------------------ typing
def normalise(name, spec):
    """Expand a typed question into (instructions, option_texts, meta).

    `noul` and `score` are sugar over a choice, exactly as the published API
    presents them: a yes/no proposition and an ordered scale.
    """
    kind = spec.get("type", "choice")
    instr = spec["instructions"]
    if kind == "choice":
        # criteria: map<string, string|null>. A null description means the key
        # itself carries the meaning.
        crit = spec["criteria"]
        keys = list(crit)
        texts = [crit[k] if crit[k] is not None else str(k) for k in keys]
        return instr, texts, {"kind": kind, "keys": keys}
    if kind == "noul":
        # criteria optional: {"true": ..., "false": ...}
        crit = spec.get("criteria") or {}
        t = crit.get("true") or "the statement is true of this state"
        f = crit.get("false") or "the statement is not true of this state"
        return instr, [t, f], {"kind": kind, "keys": [True, False]}
    if kind == "score":
        # criteria: ORDERED array of 2..10 level descriptions, 0-indexed
        levels = spec["criteria"]
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
            raise ValueError(f"question {name!r}: score criteria must be a list "
                             f"of 2 to 10 level descriptions")
        texts = [lv if isinstance(lv, str) else json.dumps(lv) for lv in levels]
        return instr, texts, {"kind": kind, "keys": list(range(len(levels))),
                              "legend": {str(i): lv
                                         for i, lv in enumerate(levels)}}
    raise ValueError(f"question {name!r}: unknown type {kind!r}")


# ------------------------------------------------------------------ packing
def pack_request(tok, state, questions):
    state_text = STATE_TMPL.format(state=state)
    s_ids = tok(state_text, add_special_tokens=False)["input_ids"]
    S = len(s_ids)

    ids = list(s_ids)
    pos = list(range(S))
    plan = []
    for name, spec in questions.items():
        instr, opts, meta = normalise(name, spec)
        i_piece = INSTR_TMPL.format(instructions=instr)
        i_ids = tok(state_text + i_piece,
                    add_special_tokens=False)["input_ids"][S:]
        instr_a = len(ids)
        ids += i_ids
        pos += list(range(S, S + len(i_ids)))
        instr_b = len(ids)
        Qi = len(i_ids)

        base_txt = state_text + i_piece
        base_n = len(tok(base_txt, add_special_tokens=False)["input_ids"])
        spans, last = [], []
        omax = 0
        for o in opts:
            o_ids = tok(base_txt + OPT_TMPL.format(text=o),
                        add_special_tokens=False)["input_ids"][base_n:]
            a = len(ids)
            ids += o_ids
            pos += list(range(S + Qi, S + Qi + len(o_ids)))
            spans.append((a, a + len(o_ids)))
            last.append(a + len(o_ids) - 1)
            omax = max(omax, len(o_ids))
        t_ids = tok(TAIL, add_special_tokens=False)["input_ids"]
        tail_a = len(ids)
        ids += t_ids
        pos += list(range(S + Qi + omax, S + Qi + omax + len(t_ids)))
        plan.append({"name": name, "meta": meta, "instr": (instr_a, instr_b),
                     "opts": spans, "opt_last": last,
                     "tail": (tail_a, len(ids)), "dec": len(ids) - 1,
                     "branch_tokens": S + Qi + omax + len(t_ids)})

    T = len(ids)
    mask = torch.zeros(T, T, dtype=torch.bool)
    for t in range(S):
        mask[t, :t + 1] = True
    for p in plan:
        ia, ib = p["instr"]
        mask[ia:ib, :S] = True
        for t in range(ia, ib):
            mask[t, ia:t + 1] = True
        for (a, b) in p["opts"]:
            mask[a:b, :S] = True
            mask[a:b, ia:ib] = True
            for t in range(a, b):
                mask[t, a:t + 1] = True
        ta, tb = p["tail"]
        mask[ta:tb, :S] = True
        mask[ta:tb, ia:ib] = True
        for (a, b) in p["opts"]:
            mask[ta:tb, a:b] = True
        for t in range(ta, tb):
            mask[t, ta:t + 1] = True
    return torch.tensor(ids), torch.tensor(pos), mask, plan, S


# ------------------------------------------------------------- confidence
def choice_confidence(probs):
    """TypeSafe's adapter formula: distance of the leading answer above uniform.

    Deliberately NOT a learned estimate that the answer is correct — keeping the
    trained distribution and this summary separate is what stops a concentrated
    distribution being mistaken for a reliable one.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    return (max(probs) - 1.0 / k) / (1.0 - 1.0 / k)


def score_confidence(probs, levels):
    """Score confidence as concentration around the modal level.

    The published adapter uses "a different formula reflecting distance from the
    modal level"; the exact form is not public, so this is a reconstruction, not
    parity. Mean absolute deviation from the mode, normalised by the worst case
    (all mass at the far end) and inverted.
    """
    if len(levels) <= 1:
        return 1.0
    m = levels[max(range(len(probs)), key=lambda i: probs[i])]
    mad = sum(p * abs(v - m) for p, v in zip(probs, levels))
    worst = max(abs(levels[0] - m), abs(levels[-1] - m)) or 1.0
    return max(0.0, 1.0 - mad / worst)


# ------------------------------------------------------- label-token readout
def _format(meta, probs):
    """Shape one answer per the published schema, shared by both readouts."""
    keys = meta["keys"]
    best = max(range(len(probs)), key=lambda i: probs[i])
    if meta["kind"] == "noul":
        return {"type": "noul", "noul": probs[0]}
    if meta["kind"] == "score":
        return {"type": "score",
                "score": sum(v * q for v, q in zip(keys, probs)),
                "confidence": score_confidence(probs, keys),
                "legend": meta["legend"],
                "probabilities": {str(k): q for k, q in zip(keys, probs)}}
    return {"type": "choice", "choice": keys[best],
            "probabilities": {str(k): q for k, q in zip(keys, probs)},
            "confidence": choice_confidence(probs)}


# ------------------------------------------------------------------ decide
@torch.no_grad()
def decide(model, head, tok, request, device=None, dtype=None):
    state = request["state"]
    questions = request["questions"]
    device = device or next(model.parameters()).device
    dtype = dtype or next(model.parameters()).dtype

    ids, pos, mask, plan, S = pack_request(tok, state, questions)
    T = len(ids)
    over = [p["name"] for p in plan if p["branch_tokens"] > BRANCH_LIMIT]
    if over:
        raise ValueError(f"branch token limit {BRANCH_LIMIT} exceeded by {over}")
    if T > REQUEST_LIMIT:
        raise ValueError(f"request token limit {REQUEST_LIMIT} exceeded ({T})")

    m = torch.zeros(mask.shape, dtype=dtype)
    m.masked_fill_(~mask, torch.finfo(dtype).min)
    out = model(input_ids=ids[None].to(device),
                position_ids=pos[None].to(device),
                attention_mask=m[None, None].to(device),
                output_hidden_states=True)
    h = out.hidden_states[-1][0]

    answers = {}
    for p in plan:
        h_dec = h[p["dec"]].float()[None]
        h_opt = h[torch.tensor(p["opt_last"])].float()[None]
        z = head(h_dec, h_opt)[0]
        probs = F.softmax(z, dim=-1).tolist()
        meta = p["meta"]
        keys = meta["keys"]
        best = max(range(len(probs)), key=lambda i: probs[i])
        if meta["kind"] == "noul":
            # published schema: {"type":"noul","noul":p} -- and noul answers
            # deliberately carry NO confidence field
            answers[p["name"]] = {"type": "noul", "noul": probs[0]}
        elif meta["kind"] == "score":
            # `score` is the probability-weighted position on the 0-indexed
            # level spectrum, not the modal level
            answers[p["name"]] = {
                "type": "score",
                "score": sum(v * q for v, q in zip(keys, probs)),
                "confidence": score_confidence(probs, keys),
                "legend": meta["legend"],
                "probabilities": {str(k): q for k, q in zip(keys, probs)},
            }
        else:
            answers[p["name"]] = {
                "type": "choice",
                "choice": keys[best],
                "probabilities": {str(k): q for k, q in zip(keys, probs)},
                "confidence": choice_confidence(probs),
            }
    body = {"model": request.get("model", "jev-local"), "answers": answers}
    body["usage"] = {
        "input_tokens": T,
        "state_tokens": S,
        # serialised size, as in the published API -- NOT neural decoding steps,
        # of which there are zero
        "output_tokens": len(tok(json.dumps(answers),
                                 add_special_tokens=False)["input_ids"]),
    }
    return body


# ------------------------------------------------------------------ checks
DEMO = {
    "model": "jev-local",
    "state": ("My payouts have failed three times. The bank says everything is "
              "fine. I have contacted support twice with no reply and I am "
              "considering closing the account."),
    "questions": {
        "queue": {"type": "choice",
                  "instructions": "Which team should handle this ticket?",
                  "criteria": {"payments": "payout failures and payment processing",
                               "account": "login and account access",
                               "other": "something else"}},
        "escalate": {"type": "noul",
                     "instructions": "Does this message require urgent human attention?"},
        "frustration": {"type": "score",
                        "instructions": "Rate the customer's frustration.",
                        "criteria": ["calm and matter-of-fact",
                                     "mildly annoyed",
                                     "clearly frustrated",
                                     "angry",
                                     "threatening to leave"]},
    },
}


def _comparable(ans):
    """A probability dict for any answer type, so the invariance checks can
    diff two answers uniformly. A `noul` answer reports a single probability
    under its own key rather than a `probabilities` map."""
    if "probabilities" in ans:
        return ans["probabilities"]
    return {"t": ans["noul"] if "noul" in ans else ans["score"]}


def selftest(model, head, tok, device, dtype):
    print("\n" + "=" * 92)
    print("PARITY SELF-TEST")
    print("=" * 92)

    # 1. the question identifier must not reach the model
    r1 = decide(model, head, tok, DEMO, device, dtype)
    renamed = {"state": DEMO["state"],
               "questions": {("zzz_" + k * 3): v
                             for k, v in DEMO["questions"].items()}}
    r2 = decide(model, head, tok, renamed, device, dtype)
    worst = 0.0
    for (ka, va), (kb, vb) in zip(r1["answers"].items(), r2["answers"].items()):
        pa = va.get("probabilities") or {"t": va["noul"]}
        pb = vb.get("probabilities") or {"t": vb["noul"]}
        worst = max(worst, max(abs(x - y) for x, y in zip(pa.values(), pb.values())))
    print(f"  question ids excluded from inference: max prob delta {worst:.3e} "
          f"-> {'PASS' if worst < 1e-6 else 'FAIL'}")

    # 2. answers must not depend on the order questions are listed
    rev = {"state": DEMO["state"],
           "questions": dict(reversed(list(DEMO["questions"].items())))}
    r3 = decide(model, head, tok, rev, device, dtype)
    worst = 0.0
    for k, v in r1["answers"].items():
        pa = _comparable(v)
        pb = _comparable(r3["answers"][k])
        worst = max(worst, max(abs(x - y) for x, y in zip(pa.values(), pb.values())))
    tol = 1e-6 if dtype == torch.float32 else 5e-2
    print(f"  question order does not change answers: max prob delta "
          f"{worst:.3e} -> {'PASS' if worst < tol else 'FAIL'} (tol {tol:g})")

    # 3. option order within a question must not change anything
    q = DEMO["questions"]["queue"]["criteria"]
    perm = {"state": DEMO["state"], "questions": {"queue": {
        "type": "choice", "instructions": DEMO["questions"]["queue"]["instructions"],
        "criteria": {k: q[k] for k in reversed(list(q))}}}}
    base = decide(model, head, tok,
                  {"state": DEMO["state"],
                   "questions": {"queue": DEMO["questions"]["queue"]}},
                  device, dtype)
    r4 = decide(model, head, tok, perm, device, dtype)
    worst = max(abs(base["answers"]["queue"]["probabilities"][k] -
                    r4["answers"]["queue"]["probabilities"][k]) for k in q)
    print(f"  option order does not change answers:   max prob delta "
          f"{worst:.3e} -> {'PASS' if worst < tol else 'FAIL'} (tol {tol:g})")

    # 4. the published confidence formula
    got = choice_confidence([0.8, 0.1, 0.1])
    print(f"  confidence formula (0.8, K=3) = {got:.3f} -> "
          f"{'PASS' if abs(got - 0.7) < 1e-9 else 'FAIL'} (article says 0.700)")

    # 5. limits are enforced
    try:
        decide(model, head, tok,
               {"state": "x " * 40000,
                "questions": {"q": DEMO["questions"]["escalate"]}}, device, dtype)
        print("  branch/request limits enforced: FAIL (no error raised)")
    except ValueError as e:
        print(f"  branch/request limits enforced: PASS ({str(e)[:52]}...)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/banking")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, a.adapter).to(device).eval()
    head = PointerHead(model.config.hidden_size).to(device).float()
    head.load_state_dict(torch.load(os.path.join(a.adapter, "pointer_head.pt")))
    head.eval()

    if a.demo or not a.selftest:
        import time
        t0 = time.time()
        out = decide(model, head, tok, DEMO, device, dtype)
        dt = (time.time() - t0) * 1000
        print(json.dumps(out, indent=2))
        print(f"\n  3 questions, one forward pass, {dt:.0f} ms "
              f"({out['usage']['input_tokens']} input tokens, "
              f"0 neural decoding steps)")
    if a.selftest:
        selftest(model, head, tok, device, dtype)


if __name__ == "__main__":
    main()
