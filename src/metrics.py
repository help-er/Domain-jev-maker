"""Metric suite for decision models.

Accuracy and ECE together are not enough to evaluate a decision model: accuracy
reads template memorisation as success, and ECE reads high confidence on
zero-entropy labels as calibration. What to report instead:

  * **Proper scoring rules.** NLL and Brier are minimised only by reporting the
    true conditional distribution. ECE is not a proper scoring rule and can be
    improved by hedging; report it, but never alone.
  * **Adaptive (equal-mass) ECE** alongside equal-width. With equal-width bins
    a confident model puts almost every item in the top bin, hiding everything
    inside it.
  * **Risk-coverage.** The deployment question is not "how often is it right"
    but "how much traffic can it decide unsupervised at a fixed precision."
    `coverage_at_precision` is the headline number; AURC summarises the curve.
  * **Expected cost** under an asymmetric policy, which is what calibration is
    ultimately for (an escalation that costs 1 against a miss that costs 9).

A row is a dict with at least: `correct` (bool), `conf` (float, the probability
of the chosen option), `k` (int, number of options). Rows carrying `probs`
(dict) and `gold` (str) additionally support Brier, NLL and classwise ECE.
"""
import math
from collections import defaultdict

EPS = 1e-12


# ------------------------------------------------------------------ intervals
def wilson(k, n, z=1.96):
    """Wilson score interval. Correct at the small n these probes produce, where
    the normal approximation is not (it can return bounds outside [0,1])."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def tie_aware_correct(probs, true, tol=1e-9):
    """Is the model's choice among the target's tied-maximal options?

    With soft targets, ties are common and meaningful: an item whose true
    distribution is 0.25/0.25/0.25/0.25 has no single right answer. Scoring it
    with `argmax(true)` silently breaks the tie by lowest index, which both
    inflates slot 0's apparent gold share and makes accuracy on those items a
    property of the tie-break convention rather than of the model.

    Measured on the v2 held-out set: 14.2% of items have tied maxima, and this
    rule raises accuracy by +3.0 to +5.2 points depending on the model (more for
    the better ones, so the strict rule understates their margin).
    """
    if not probs or not true:
        return False
    m = max(true.values())
    pick = max(probs, key=probs.get)
    return true.get(pick, 0.0) >= m - tol


# ------------------------------------------------------- proper scoring rules
def brier(probs, gold):
    """Multiclass Brier score, range [0, 2]. Bounded, so less tail-sensitive
    than NLL — worth reporting both, they disagree in informative ways."""
    return sum((p - (1.0 if o == gold else 0.0)) ** 2 for o, p in probs.items())


def nll(probs, gold):
    return -math.log(max(probs.get(gold, 0.0), EPS))


# ----------------------------------------------------------------- calibration
def _bin_stats(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    conf = sum(r["conf"] for r in rows) / n
    return n, acc, conf


def ece_equal_width(rows, n_bins=10):
    """Classic ECE. Bins are fixed-width, so a model concentrated near 1.0 puts
    almost everything in one bin and the number stops being informative."""
    if not rows:
        return 0.0
    total, err = len(rows), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [r for r in rows
               if r["conf"] >= lo and (r["conf"] < hi or b == n_bins - 1)]
        if sel:
            n, acc, conf = _bin_stats(sel)
            err += abs(acc - conf) * n / total
    return err


def ece_adaptive(rows, n_bins=10):
    """Equal-mass bins: every bin carries the same number of items, so a
    concentrated distribution still gets resolved."""
    if not rows:
        return 0.0
    s = sorted(rows, key=lambda r: r["conf"])
    total, err = len(s), 0.0
    edges = [round(i * total / n_bins) for i in range(n_bins + 1)]
    for a, b in zip(edges, edges[1:]):
        sel = s[a:b]
        if sel:
            n, acc, conf = _bin_stats(sel)
            err += abs(acc - conf) * n / total
    return err


def classwise_ece(rows, n_bins=10):
    """Mean adaptive ECE computed per predicted class. Aggregate calibration can
    look fine while individual classes are badly over- or under-confident in
    opposite directions."""
    by = defaultdict(list)
    for r in rows:
        by[r.get("pred", "?")].append(r)
    if not by:
        return 0.0
    return sum(ece_adaptive(v, n_bins) for v in by.values()) / len(by)


def reliability_table(rows, n_bins=10, adaptive=True):
    """Rows of (lo, hi, n, accuracy, mean_conf) for a reliability diagram."""
    if not rows:
        return []
    out = []
    if adaptive:
        s = sorted(rows, key=lambda r: r["conf"])
        edges = [round(i * len(s) / n_bins) for i in range(n_bins + 1)]
        for a, b in zip(edges, edges[1:]):
            sel = s[a:b]
            if sel:
                n, acc, conf = _bin_stats(sel)
                out.append((sel[0]["conf"], sel[-1]["conf"], n, acc, conf))
    else:
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            sel = [r for r in rows
                   if r["conf"] >= lo and (r["conf"] < hi or b == n_bins - 1)]
            if sel:
                n, acc, conf = _bin_stats(sel)
                out.append((lo, hi, n, acc, conf))
    return out


# --------------------------------------------------------------- risk-coverage
def risk_coverage(rows):
    """Curve of (coverage, selective accuracy) as the confidence threshold drops.

    Selective accuracy at coverage c = accuracy over the c most-confident items.
    This is the curve a routing policy actually rides.
    """
    if not rows:
        return []
    s = sorted(rows, key=lambda r: -r["conf"])
    out, hits = [], 0
    for i, r in enumerate(s, 1):
        hits += bool(r["correct"])
        out.append((i / len(s), hits / i))
    return out


def aurc(rows):
    """Area under the risk–coverage curve, using risk = 1 - selective accuracy.
    Lower is better; it rewards ranking errors below correct answers, which is
    exactly what a confidence signal is for."""
    curve = risk_coverage(rows)
    if not curve:
        return 0.0
    return sum(1.0 - acc for _, acc in curve) / len(curve)


def coverage_at_precision(rows, precision=0.99):
    """Largest coverage whose selective accuracy still meets `precision`.

    The headline deployment number: the share of traffic that can be decided
    without review at a fixed quality bar.
    """
    best = 0.0
    for cov, acc in risk_coverage(rows):
        if acc >= precision:
            best = cov
    return best


# ------------------------------------------------------------- decision policy
def expected_cost(rows, cost_fp=1.0, cost_fn=9.0, positive=None):
    """Expected cost of a threshold policy on a binary decision.

    Worked example: an unnecessary escalation costs 1, a missed urgent case
    costs 9, so the optimal rule escalates at p > cost_fp/(cost_fp+cost_fn) = 0.1.
    A model whose probabilities are honest minimises this; a model that is merely
    accurate does not. Returns (best_threshold, cost_at_best, cost_at_argmax).
    """
    if positive is None or not rows:
        return (None, 0.0, 0.0)

    def cost(th):
        c = 0.0
        for r in rows:
            p = r.get("probs", {}).get(positive, 0.0)
            act = r.get("gold") == positive
            c += (cost_fn if act else 0.0) if p <= th else (0.0 if act else cost_fp)
        return c / len(rows)

    ths = [i / 200 for i in range(201)]
    best = min(ths, key=cost)
    return (best, cost(best), cost(0.5))


# ------------------------------------------------------------------ reporting
def summarise(rows):
    """Every headline metric for one group of rows."""
    if not rows:
        return {}
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    wrong = [r for r in rows if not r["correct"]]
    has_probs = all("probs" in r and "gold" in r for r in rows)
    has_t = all("probs" in r and "true" in r for r in rows)
    return {
        "n": n,
        "acc": k / n,
        "acc_tie_aware": (sum(tie_aware_correct(r["probs"], r["true"])
                              for r in rows) / n if has_t else None),
        "ci": (lo, hi),
        "chance": sum(1.0 / r["k"] for r in rows) / n,
        "mean_conf": sum(r["conf"] for r in rows) / n,
        "overconf": sum(r["conf"] for r in rows) / n - k / n,
        "conf_when_wrong": (sum(r["conf"] for r in wrong) / len(wrong)
                            if wrong else None),
        "brier": (sum(brier(r["probs"], r["gold"]) for r in rows) / n
                  if has_probs else None),
        "nll": (sum(nll(r["probs"], r["gold"]) for r in rows) / n
                if has_probs else None),
        "ece_ew": ece_equal_width(rows),
        "ece_ad": ece_adaptive(rows),
        "ece_cw": classwise_ece(rows),
        "aurc": aurc(rows),
        "cov99": coverage_at_precision(rows, 0.99),
        "cov95": coverage_at_precision(rows, 0.95),
    }


HEADER = (f"{'group':<22} {'n':>4} {'acc':>6} {'95% CI':>15} {'conf':>6} "
          f"{'over':>6} {'wrong@':>7} {'Brier':>6} {'NLL':>6} "
          f"{'ECEad':>6} {'AURC':>6} {'cov95':>6} {'cov99':>6}")


def row_line(name, m):
    def f(x, w=6, p=3):
        return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{'--':>{w}}"
    return (f"{name:<22} {m['n']:>4} {m['acc']:>6.3f} "
            f"[{m['ci'][0]:>5.3f},{m['ci'][1]:>5.3f}] {m['mean_conf']:>6.3f} "
            f"{m['overconf']:>+6.3f} {f(m['conf_when_wrong'], 7)} "
            f"{f(m['brier'])} {f(m['nll'])} {f(m['ece_ad'])} {f(m['aurc'])} "
            f"{m['cov95']:>6.0%} {m['cov99']:>6.0%}")


def report(groups, title):
    """groups: {name: rows}"""
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
    print(HEADER)
    for name, rows in groups.items():
        m = summarise(rows)
        if m:
            print(row_line(name, m))
