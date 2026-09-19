"""Build a domain decision corpus from a spec file plus real labelled text.

The corpus has three kinds of item and needs all three:

  1. **In-domain, one-hot** -- real human-written utterances with their real
     labels. This is where the domain knowledge comes from.
  2. **Out-of-scope, one-hot** -- messages this assistant should refuse to
     route, drawn from the corpus's own out-of-scope class and from in-scope
     intents of *other* domains. This is the only thing that teaches
     abstention.
  3. **Genuinely ambiguous, soft** -- two constructions, both built from real
     utterances so the uncertainty is visible in the text:
       * *composed*: join two real utterances from a confusable pair
         ("What's my balance? Also, when is my bill due?") -> 50/50.
       * *hedged*: the sender names both intents and says they are unsure
         which, leaning to the one stated first -> 67/33.

Usage:
  python src/gen_domain.py --spec domains/banking.json --out data/banking
"""
import argparse
import json
import os
import random

SEED = 20260918


def load_rows(source):
    """Return [(text, label), ...].

    Two source kinds are supported out of the box. To use your own corpus,
    add a branch here that returns the same pairs -- nothing else changes.
    """
    kind = source.get("kind", "parquet")
    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(source["dataset"], source.get("config"),
                          split=source["split"])
        names = ds.features[source.get("label_field", "intent")].names
        tf = source.get("text_field", "text")
        lf = source.get("label_field", "intent")
        return [(str(r[tf]), names[int(r[lf])]) for r in ds]

    # parquet written by `datasets`, label column stored as a class index
    import pandas as pd
    import pyarrow.parquet as pq
    path = source["path"]
    tf = source.get("text_field", "text")
    lf = source.get("label_field", "intent")
    md = pq.read_table(path).schema.metadata or {}
    key = [k for k in md if b"huggingface" in k.lower()]
    df = pd.read_parquet(path)
    if key:
        names = json.loads(md[key[0]].decode())["info"]["features"][lf]["names"]
        return [(str(t), names[int(i)]) for t, i in zip(df[tf], df[lf])]
    return [(str(t), str(i)) for t, i in zip(df[tf], df[lf])]


def render(rng, spec, labels, soft):
    """One record, with option order randomised per item.

    Randomising here is what keeps the model from learning a position prior;
    it is the largest single accuracy lever in the recipe.
    """
    order = list(range(len(labels)))
    rng.shuffle(order)
    crit = dict(spec["criteria"])
    display = [labels[i] for i in order]
    return {
        "question": rng.choice(spec["question"]),
        "options": display,
        "rendered": "; ".join(f"{labels[i]} = {crit[labels[i]]}" for i in order),
        "criteria": crit,
        "vocab_map": {l: l for l in labels},
        "soft": {l: soft.get(l, 0.0) for l in labels},
        "gold": max(soft, key=soft.get),
    }


def build(spec, rows, rng, n_target, soft_frac):
    intents = list(spec["intents"])
    abstain = spec["abstain_label"]
    labels = intents + [abstain]
    spec = dict(spec)
    spec["criteria"] = dict(spec["intents"])
    spec["criteria"][abstain] = spec["abstain_criteria"]
    oos_name = spec.get("oos_label", "oos")

    in_dom, oos_own, other = [], [], []
    for text, name in rows:
        if name in spec["intents"]:
            in_dom.append((text, name))
        elif name == oos_name:
            oos_own.append(text)
        else:
            other.append(text)       # another domain's intent: out of scope here

    by_intent = {}
    for t, n in in_dom:
        by_intent.setdefault(n, []).append(t)

    n_hard_want = int(n_target * (1 - soft_frac))
    n_abstain = n_hard_want // 4     # a quarter of the hard items teach abstention
    n_in = n_hard_want - n_abstain

    out = []
    # 1. in-domain, balanced across intents and capped by what exists
    per = max(1, n_in // len(intents))
    for name in intents:
        pool = list(by_intent.get(name, []))
        rng.shuffle(pool)
        for t in pool[:per]:
            out.append({"text": t, "kind": "in_domain",
                        **render(rng, spec, labels, {name: 1.0})})

    # 2. out-of-scope. Use the corpus's own out-of-scope class first -- those
    #    are the genuine "no supported intent" cases -- then top up from other
    #    domains' intents.
    rng.shuffle(oos_own)
    rng.shuffle(other)
    take_own = min(len(oos_own), max(1, n_abstain // 2))
    picked = ([(t, "oos") for t in oos_own[:take_own]] +
              [(t, "other_domain") for t in other[:n_abstain - take_own]])
    rng.shuffle(picked)
    for t, src in picked:
        out.append({"text": t, "kind": f"abstain_{src}",
                    **render(rng, spec, labels, {abstain: 1.0})})

    # Derive the soft count from the hard items ACTUALLY built. Corpora cap the
    # number of utterances per intent, so asking for more silently shrinks the
    # hard half and would inflate the soft fraction.
    n_hard = len(out)
    n_soft = int(round(n_hard * soft_frac / max(1e-9, 1 - soft_frac)))

    # 3. genuinely ambiguous, soft targets
    pairs = [p for p in spec["confusable"]
             if p[0] in by_intent and p[1] in by_intent]
    made = 0
    while made < n_soft and pairs:
        a, b = rng.choice(pairs)
        ta, tb = rng.choice(by_intent[a]), rng.choice(by_intent[b])
        qa = ta.rstrip(" .?") + "?"
        if rng.random() < 0.75:
            text = rng.choice(spec["joiners"]).format(a=qa, b=tb)
            soft = {a: 0.5, b: 0.5}
        else:
            text = rng.choice(spec["hedges"]).format(
                a=qa, b=tb.rstrip(" .?") + "?")
            soft = {a: 0.67, b: 0.33}
        out.append({"text": text, "kind": "ambiguous",
                    **render(rng, spec, labels, soft)})
        made += 1

    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--soft-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    spec = json.load(open(a.spec, encoding="utf-8"))
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    for name, n, src in (("train", a.n_train, spec["source"]["train"]),
                         ("heldout", a.n_test, spec["source"]["test"])):
        items = build(spec, load_rows(src), rng, n, a.soft_frac)
        p = os.path.join(a.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        kinds = Counter(r["kind"] for r in items)
        soft = sum(1 for r in items if max(r["soft"].values()) < 0.999)
        print(f"{name}: {len(items)} -> {p}")
        print(f"  kinds: {dict(kinds)}")
        print(f"  soft-target: {soft} ({soft/len(items):.1%})   "
              f"options per item: {len(spec['intents'])+1}")

    tr = {json.loads(l)["text"]
          for l in open(os.path.join(a.out, "train.jsonl"), encoding="utf-8")}
    te = {json.loads(l)["text"]
          for l in open(os.path.join(a.out, "heldout.jsonl"), encoding="utf-8")}
    overlap = len(tr & te)
    print(f"\ntrain/test verbatim text overlap: {overlap}")
    if overlap:
        raise SystemExit("train and test share text; check your source splits")


if __name__ == "__main__":
    main()
