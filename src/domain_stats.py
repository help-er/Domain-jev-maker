"""Paired significance tests for one domain: local model vs hosted API.

Both systems answer the same items, so the comparison is paired: McNemar for
accuracy, a paired bootstrap for KL. Two independently-computed accuracies with
overlapping intervals say nothing about a paired difference.

Scores at matched precision by default; see `ab_domain.requantise`.

Usage:
  python src/domain_stats.py --api out/api_banking.jsonl \
      --local out/local_banking.jsonl --tag BANKING
"""
import argparse
import json
import math
import random

from ab_domain import requantise

ABSTAIN = "out_of_scope"


def mcnemar(b, c):
    """Continuity-corrected chi-square on the discordant pairs, 1 df."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return chi, math.erfc(math.sqrt(chi / 2))


def boot_diff(vals, iters=4000, seed=7):
    """Paired bootstrap CI on mean(local - api)."""
    rng = random.Random(seed)
    n = len(vals)
    out = [sum(vals[rng.randrange(n)] for _ in range(n)) / n
           for _ in range(iters)]
    out.sort()
    return out[int(0.025 * iters)], out[int(0.975 * iters)]


def pear(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="scored rows from ab_domain")
    ap.add_argument("--local", required=True, help="rows from score_domain")
    ap.add_argument("--tag", default="DOMAIN")
    ap.add_argument("--quant", type=int, default=2)
    ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args()

    def rd(p):
        rows = (json.loads(l) for l in open(p, encoding="utf-8"))
        return [requantise(r, a.quant, a.floor) if a.quant else r for r in rows]

    A = {r["i"]: r for r in rd(a.api)}
    L = [r for r in rd(a.local) if r["i"] in A]

    label = f"matched {a.quant} dp" if a.quant else "raw precision"
    print(f"\n{a.tag}: paired local-vs-API, {label}, n={len(L)}")
    print("-" * 96)

    for subset, keep in (("all items", lambda r: True),
                         ("determinate", lambda r: r["task"] != "ambiguous"),
                         ("ambiguous", lambda r: r["task"] == "ambiguous")):
        rows = [r for r in L if keep(r)]
        if not rows:
            continue
        b = sum(1 for r in rows if r["correct"] and not A[r["i"]]["correct"])
        c = sum(1 for r in rows if not r["correct"] and A[r["i"]]["correct"])
        _, p = mcnemar(b, c)
        la = sum(r["correct"] for r in rows) / len(rows)
        aa = sum(A[r["i"]]["correct"] for r in rows) / len(rows)
        lo, hi = boot_diff([r["correct"] - A[r["i"]]["correct"] for r in rows])
        print(f"  acc  {subset:<12} n={len(rows):>4}  local {la:.4f}  "
              f"api {aa:.4f}  diff {la-aa:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"McNemar b={b} c={c} p={p:.3f}  "
              f"{'significant' if p < 0.05 else 'n.s.'}")

    for metric in ("kl", "ce"):
        d = [r[metric] - A[r["i"]][metric] for r in L]
        lo, hi = boot_diff(d)
        lm = sum(r[metric] for r in L) / len(L)
        am = sum(A[r["i"]][metric] for r in L) / len(L)
        sig = "significant" if (lo < 0) == (hi < 0) else "n.s."
        print(f"  {metric.upper():<4} {'all items':<12} n={len(L):>4}  "
              f"local {lm:.4f}  api {am:.4f}  diff {lm-am:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]{'':<26}{sig}")

    rl = pear([r["h_true"] for r in L], [r["h_pred"] for r in L])
    ra = pear([A[r["i"]]["h_true"] for r in L], [A[r["i"]]["h_pred"] for r in L])
    print(f"  r(H_true,H_pred)          local {rl:+.4f}  api {ra:+.4f}  "
          f"diff {rl-ra:+.4f}")

    for name, rows in (("local", L), ("api", [A[r["i"]] for r in L])):
        tp = sum(1 for x in rows if x["pred"] == ABSTAIN and x["gold"] == ABSTAIN)
        fp = sum(1 for x in rows if x["pred"] == ABSTAIN and x["gold"] != ABSTAIN)
        fn = sum(1 for x in rows if x["pred"] != ABSTAIN and x["gold"] == ABSTAIN)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"  abstain {name:<6} precision {prec:.4f}  recall {rec:.4f}  "
              f"F1 {2*prec*rec/max(1e-9, prec+rec):.4f}")


if __name__ == "__main__":
    main()
