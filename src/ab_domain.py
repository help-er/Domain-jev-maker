"""Compare a local domain model against a hosted decision API on identical items.

Written against TypeSafe's Jev (`POST /v1/systemone`), but the request shape is
small and the scoring is generic.

Two things this handles that a naive comparison does not:

  * **Matched precision.** The API reports probabilities rounded to two
    decimals, so values below 0.005 arrive as zero. An unguarded KL against
    such a response is determined by the epsilon you pick, not by the model.
    Both sides are therefore rounded to the same precision, zeros are replaced
    by the midpoint of the interval they are known to lie in, and each row is
    renormalised. `--floor-sweep` shows the sensitivity to that choice.
  * **A fair prompt for the API.** `--fewshot N` puts N labelled examples from
    the training split into the API's `state`. Compare against this, not
    against the zero-shot arm, before drawing conclusions about accuracy.

Usage:
  python src/ab_domain.py --data-dir data/banking --dry-run
  python src/ab_domain.py --data-dir data/banking --key-file ~/.ts_key
  python src/ab_domain.py --data-dir data/banking --fewshot 31 --limit 300 \
      --key-file ~/.ts_key
  python src/ab_domain.py --data-dir data/banking --compare-only --floor-sweep
"""
import argparse
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import metrics as M

ENDPOINT = os.environ.get("DECISION_API", "https://api.typesafe.ai/v1/systemone")
MODEL = os.environ.get("DECISION_MODEL", "jev-latest")
EPS = 1e-12
ABSTAIN = "out_of_scope"
KINDS = ["in_domain", "abstain_oos", "abstain_other_domain", "ambiguous"]

FEWSHOT_PREFIX = None


def read_key(key_file=None):
    """Key from a file or the environment; never written into the repo."""
    if key_file and os.path.exists(os.path.expanduser(key_file)):
        return open(os.path.expanduser(key_file), encoding="utf-8").read().strip()
    return os.environ.get("DECISION_API_KEY") or os.environ.get("TYPESAFE_API_KEY")


def post(req, key, timeout=60, max_retries=6):
    body = json.dumps(req).encode()
    r = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    delay = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError("exhausted retries")


def build_fewshot(train_path, n, seed=11):
    """One labelled example per intent, from the TRAINING split."""
    rng = random.Random(seed)
    recs = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    by = {}
    for r in recs:
        if max(r["soft"].values()) > 0.999:
            by.setdefault(max(r["soft"], key=r["soft"].get), []).append(r)
    picks = [rng.choice(by[lab]) for lab in sorted(by)]
    rng.shuffle(picks)
    body = "\n".join(f'- "{r["text"]}" -> {max(r["soft"], key=r["soft"].get)}'
                     for r in picks[:n])
    return ("Examples of how past messages were categorised:\n" + body +
            "\n\nThe message to categorise is:\n")


def build_request(rec):
    """The API sees exactly what the local model sees: same options, same
    order, same criteria text, same question phrasing."""
    order = list(rec["options"])
    crit = rec["criteria"]
    state = rec["text"] if not FEWSHOT_PREFIX else FEWSHOT_PREFIX + rec["text"]
    return {
        "model": MODEL,
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": rec["question"],
                "criteria": {c: crit[c] for c in order},
            }
        },
    }, order


def fetch(recs, cache, key, workers=4):
    done = set()
    if os.path.exists(cache):
        for line in open(cache, encoding="utf-8"):
            done.add(json.loads(line)["i"])
    todo = [i for i in range(len(recs)) if i not in done]
    if not todo:
        print(f"all {len(recs)} items already cached")
        return
    print(f"fetching {len(todo)} items ({len(done)} cached), {workers} workers")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    lock = threading.Lock()
    fh = open(cache, "a", encoding="utf-8")
    n_done = [0]
    stop = threading.Event()

    def one(i):
        if stop.is_set():
            return
        req, order = build_request(recs[i])
        try:
            resp = post(req, key)
        except Exception as e:                              # noqa: BLE001
            with lock:
                print(f"  item {i}: {e}", flush=True)
            stop.set()
            return
        with lock:
            fh.write(json.dumps({"i": i, "canon_order": order,
                                 "response": resp}) + "\n")
            fh.flush()
            n_done[0] += 1
            if n_done[0] % 50 == 0:
                print(f"  {n_done[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    fh.close()


def score_cache(recs, cache, out):
    rows = []
    for line in open(cache, encoding="utf-8"):
        c = json.loads(line)
        rec = recs[c["i"]]
        ans = c["response"]["answers"]["decision"]
        order = c["canon_order"]
        probs = ans.get("probabilities") or {}
        p = [float(probs.get(k, 0.0)) for k in order]
        s = sum(p) or 1.0
        p = [x / s for x in p]
        q = [rec["soft"][k] for k in order]
        s = sum(q) or 1.0
        q = [x / s for x in q]
        k = len(order)
        ce = -sum(a * math.log(max(b, EPS)) for a, b in zip(q, p))
        hq = -sum(x * math.log(max(x, EPS)) for x in q if x > 0)
        hp = -sum(x * math.log(max(x, EPS)) for x in p if x > 0)
        am = max(range(k), key=lambda j: p[j])
        aq = max(range(k), key=lambda j: q[j])
        rows.append({"i": c["i"], "task": rec["kind"], "k": k,
                     "correct": am == aq, "conf": max(p),
                     "pred": order[am], "gold": order[aq],
                     "probs": {order[j]: p[j] for j in range(k)},
                     "true": {order[j]: q[j] for j in range(k)},
                     "kl": ce - hq, "ce": ce, "h_true": hq, "h_pred": hp,
                     "api_confidence": ans.get("confidence")})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"scored {len(rows)} API responses -> {out}")
    return rows


def requantise(row, nd=2, floor=None):
    """Round a row to `nd` decimals and replace zeros with a stated floor.

    A value reported as 0.00 is known only to lie in [0, 10^-nd / 2), so the
    default floor is the midpoint of that interval. Apply this to BOTH systems
    or the comparison measures response formatting.
    """
    step = 10.0 ** (-nd)
    floor = step / 4 if floor is None else floor
    keys = list(row["probs"])
    p = [max(round(row["probs"][k] / step) * step, floor) for k in keys]
    s = sum(p) or 1.0
    p = [x / s for x in p]
    q = [row["true"][k] for k in keys]
    s = sum(q) or 1.0
    q = [x / s for x in q]
    ce = -sum(a * math.log(max(b, EPS)) for a, b in zip(q, p))
    hq = -sum(x * math.log(max(x, EPS)) for x in q if x > 0)
    hp = -sum(x * math.log(max(x, EPS)) for x in p if x > 0)
    am = max(range(len(keys)), key=lambda j: p[j])
    aq = max(range(len(keys)), key=lambda j: q[j])
    out = dict(row)
    out.update({"probs": dict(zip(keys, p)), "true": dict(zip(keys, q)),
                "conf": max(p), "pred": keys[am], "gold": keys[aq],
                "correct": am == aq, "kl": ce - hq, "ce": ce,
                "h_true": hq, "h_pred": hp})
    return out


def pear(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def load(p, only, nd=None, floor=None):
    if not os.path.exists(p):
        return None
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    if only is not None:
        rs = [r for r in rs if r["i"] in only]
    if nd:
        rs = [requantise(r, nd, floor) for r in rs]
    return rs or None


def table(paths, only=None, nd=2, floor=None, title="DOMAIN"):
    print("\n" + "=" * 104)
    print(f"{title}: local model vs hosted API, identical items")
    if nd:
        f = (10.0 ** -nd) / 4 if floor is None else floor
        print(f"both sides quantised to {nd} dp, zeros floored at {f:g} "
              f"and renormalised")
    else:
        print("raw outputs, precision not matched")
    if only is not None:
        print(f"({len(only)} items fetched from the API)")
    print("=" * 104)
    print(f"{'model':<22} {'n':>5} {'KL':>8} {'acc':>8} {'acc(tie)':>9} "
          f"{'r(H_t,H_p)':>11} {'H_pred':>8} {'ECEad':>8} {'AURC':>8} "
          f"{'abstF1':>8}")
    for name, p in paths:
        rs = load(p, only, nd, floor)
        if rs is None:
            continue
        mm = M.summarise(rs)
        n = len(rs)
        tp = sum(1 for x in rs if x["pred"] == ABSTAIN and x["gold"] == ABSTAIN)
        fp = sum(1 for x in rs if x["pred"] == ABSTAIN and x["gold"] != ABSTAIN)
        fn = sum(1 for x in rs if x["pred"] != ABSTAIN and x["gold"] == ABSTAIN)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"{name:<22} {n:>5} {sum(x['kl'] for x in rs)/n:>8.4f} "
              f"{mm['acc']:>8.4f} {mm['acc_tie_aware']:>9.4f} "
              f"{pear([x['h_true'] for x in rs], [x['h_pred'] for x in rs]):>+11.4f} "
              f"{sum(x['h_pred'] for x in rs)/n:>8.4f} {mm['ece_ad']:>8.4f} "
              f"{mm['aurc']:>8.4f} {f1:>8.4f}")

    print(f"\n{'model':<22} " + " ".join(f"{k:>21}" for k in KINDS))
    for name, p in paths:
        rs = load(p, only, nd, floor)
        if rs is None:
            continue
        cells = []
        for k in KINDS:
            sub = [x for x in rs if x["task"] == k]
            if not sub:
                cells.append(f"{'-':>21}")
                continue
            cells.append(f"{sum(1 for x in sub if x['correct'])/len(sub):>9.3f}"
                         f" KL{sum(x['kl'] for x in sub)/len(sub):>8.3f}")
        print(f"{name:<22} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--local", default=None,
                    help="scored rows from score_domain.py")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--quant", type=int, default=2)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--floor-sweep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    dom = os.path.basename(os.path.normpath(a.data_dir))
    recs = [json.loads(l) for l in
            open(os.path.join(a.data_dir, a.split + ".jsonl"), encoding="utf-8")]

    global FEWSHOT_PREFIX
    if a.fewshot:
        FEWSHOT_PREFIX = build_fewshot(os.path.join(a.data_dir, "train.jsonl"),
                                       a.fewshot)
    # every artifact path carries the domain and the arm, so a second domain
    # or a second arm cannot overwrite the first
    suffix = f"_fs{a.fewshot}" if a.fewshot else ""
    cache = os.path.join(a.out_dir, f"api_cache_{dom}{suffix}.jsonl")
    api_rows = os.path.join(a.out_dir, f"api_{dom}{suffix}.jsonl")
    local_rows = a.local or os.path.join(a.out_dir, f"local_{dom}.jsonl")

    if a.dry_run:
        req, order = build_request(recs[0])
        print(json.dumps(req, indent=2)[:1800])
        n = a.limit or len(recs)
        print(f"\n-> {n} requests like this ({len(order)} options each)")
        return

    if not a.compare_only:
        key = read_key(a.key_file)
        if not key:
            print("no API key: pass --key-file or set DECISION_API_KEY")
            return
        fetch(recs[:a.limit] if a.limit else recs, cache, key, a.workers)

    only = None
    if os.path.exists(cache):
        only = {r["i"] for r in score_cache(recs, cache, api_rows)}
    paths = [(MODEL, api_rows), (f"local-{dom}", local_rows)]
    title = a.title or dom.upper()
    table(paths, only, a.quant, a.floor, title)
    if a.quant:
        table(paths, only, 0, None, title + " (raw, unmatched precision)")
    if a.floor_sweep:
        print()
        print("  floor sensitivity (mean KL):")
        print(f"  {'floor':>10} " + " ".join(f"{n:>22}" for n, _ in paths))
        for fl in (0.005, 0.0025, 0.001, 1e-4, 1e-12):
            cells = []
            for _, p in paths:
                rs = load(p, only, a.quant or 2, fl)
                cells.append(f"{'-':>22}" if rs is None else
                             f"{sum(x['kl'] for x in rs)/len(rs):>22.4f}")
            print(f"  {fl:>10g} " + " ".join(cells))


if __name__ == "__main__":
    main()
