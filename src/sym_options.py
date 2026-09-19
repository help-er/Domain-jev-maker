"""Order-invariant option encoding.

A pointer readout over a plain causal layout is not order-invariant: it reads
each option's span-final hidden state, but option 3 has attended to options 1-2
while option 1 has not, so the option representations are not exchangeable and
the pointer inherits that asymmetry.

The fix is to encode every option as an **isolated branch over a shared
prefix**:

    [ state + question prefix ][ opt 1 ][ opt 2 ] ... [ opt K ][ decision tail ]
      positions 0..P-1           P..P+L1   P..P+L2      P..P+LK   P+Lmax..

    prefix token   -> keys 0..t                      (causal)
    option-i token -> prefix + own span, causally    (never a sibling option)
    tail token     -> prefix + ALL option spans + own tail

Every option then occupies the *same positions* and sees the *same context*, so
the option encodings are exchangeable by construction. The decision tail sees
all of them, so the readout is still listwise. The result is exactly
permutation-invariant, which `check_invariance()` asserts rather than assumes,
and it costs nothing: the packed sequence is the same length as the ordinary
one.

The trade: options cannot see each other, so an option like "none of the above"
cannot condition on its siblings. The decision position still reads the whole
list, so joint effects survive there.
"""
import math

import torch

from config import BASE, SYSTEM

PREFIX_TMPL = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
               "<|im_start|>user\n"
               "State:\n{state}\n\n"
               "Question: {question}\n"
               "Allowed options:\n")
OPT_TMPL = "- {text}\n"
TAIL = "Decision:<|im_end|>\n<|im_start|>assistant\n"


def pack_options(tok, state, question, options):
    """Pack prefix + isolated option branches + decision tail.

    Returns ids, position_ids, bool mask (T,T), option-final indices, decision
    index. Token boundaries are found by tokenising each growing prefix and
    verifying containment, the same discipline used elsewhere in the repo.
    """
    prefix = PREFIX_TMPL.format(state=state, question=question)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    P = len(p_ids)

    opt_ids = []
    for o in options:
        piece = OPT_TMPL.format(text=o)
        full = tok(prefix + piece, add_special_tokens=False)["input_ids"]
        assert full[:P] == p_ids, "option boundary moved the prefix tokenisation"
        opt_ids.append(full[P:])
    t_ids = tok(TAIL, add_special_tokens=False)["input_ids"]

    ids = list(p_ids)
    pos = list(range(P))
    spans, opt_last = [], []
    Lmax = max(len(o) for o in opt_ids)
    for o in opt_ids:
        a = len(ids)
        ids += o
        pos += list(range(P, P + len(o)))      # every option starts at P
        spans.append((a, a + len(o)))
        opt_last.append(a + len(o) - 1)
    tail_a = len(ids)
    ids += t_ids
    pos += list(range(P + Lmax, P + Lmax + len(t_ids)))
    dec = len(ids) - 1
    T = len(ids)

    mask = torch.zeros(T, T, dtype=torch.bool)
    for t in range(P):                          # prefix: ordinary causal
        mask[t, :t + 1] = True
    for (a, b) in spans:                        # options: prefix + self only
        mask[a:b, :P] = True
        for t in range(a, b):
            mask[t, a:t + 1] = True
    mask[tail_a:T, :P] = True                   # tail: prefix + every option
    for (a, b) in spans:
        mask[tail_a:T, a:b] = True
    for t in range(tail_a, T):                  # tail: causal within itself
        mask[t, tail_a:t + 1] = True
    return torch.tensor(ids), torch.tensor(pos), mask, opt_last, dec


def to_additive(mask, dtype):
    m = torch.zeros(mask.shape, dtype=dtype)
    m.masked_fill_(~mask, torch.finfo(dtype).min)
    return m[None, None]


@torch.no_grad()
def hidden_for(model, tok, state, question, options, device):
    ids, pos, mask, opt_last, dec = pack_options(tok, state, question, options)
    dtype = next(model.parameters()).dtype
    out = model(input_ids=ids[None].to(device),
                position_ids=pos[None].to(device),
                attention_mask=to_additive(mask, dtype).to(device),
                output_hidden_states=True)
    h = out.hidden_states[-1][0]
    return h[torch.tensor(opt_last)], h[dec]


def check_invariance(model, tok, device):
    """Permuting the options must leave every option's representation and the
    decision representation unchanged, up to numerical noise."""
    print("\n" + "=" * 92)
    print("PERMUTATION INVARIANCE of the symmetric option encoding")
    print("=" * 92)
    state = ("Checkout fails completely. Every customer is affected. "
             "No workaround has been found.")
    q = "What is the priority of this support ticket?"
    opts = [
        "cosmetic or non-blocking, and can wait for a future sprint",
        "affects a few users, or there is a usable workaround",
        "many users affected, or a core flow is badly degraded",
        "complete outage, data loss in progress, or an active security incident",
    ]
    ho, hd = hidden_for(model, tok, state, q, opts, device)
    worst_o = worst_d = 0.0
    for pm in ([3, 1, 2, 0], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]):
        ho2, hd2 = hidden_for(model, tok, state, q, [opts[i] for i in pm], device)
        back = torch.zeros_like(ho2)
        for j, src in enumerate(pm):
            back[src] = ho2[j]
        worst_o = max(worst_o, (ho.float() - back.float()).abs().max().item())
        worst_d = max(worst_d, (hd.float() - hd2.float()).abs().max().item())
    dt = next(model.parameters()).dtype
    tol = 2e-3 if dt == torch.float32 else 0.5
    print(f"  dtype {dt}")
    print(f"  max abs diff, option reps: {worst_o:.3e}")
    print(f"  max abs diff, decision rep: {worst_d:.3e}")
    ok = worst_o < tol and worst_d < tol
    print(f"  -> {'PASS' if ok else 'FAIL'} (tolerance {tol:g})")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model.to(device).eval()
    check_invariance(model, tok, device)
