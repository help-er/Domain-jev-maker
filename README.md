# Domain decision models

Train a small, well-calibrated **decision model** for one domain: it takes a
shared state and a typed question and returns a probability distribution over
options. No text is generated — the distribution is the answer.

One LoRA adapter over a 1.5B instruct backbone, about an hour on a single
consumer GPU, served behind a `POST /v1/systemone` contract.

**[GUIDE.md](GUIDE.md)** — the recipe. 
**[RESULTS.md](RESULTS.md)** — what it measures against a frontier hosted model.

To use this, simply download the files and point your favorite agent at them with your intended domain. Read on for more info.

---

## The finding

Measured on two domains (retail banking, travel logistics) against live
`jev-latest` on 1,173 identical held-out items each:

| | banking | travel |
|---|---|---|
| KL(true‖pred) — hosted / local | 0.580 / **0.168** | 0.573 / **0.124** |
| r(H_true, H_pred) — hosted / local | +0.343 / **+0.933** | +0.428 / **+0.950** |
| determinate accuracy — hosted / local | 0.952 / 0.966 | 0.936 / 0.974 |

Give the hosted model one labelled example per intent in its `state` — a prompt
change, no training — and its determinate accuracy rises to **0.982** and
**0.976**, matching or beating the local specialist (McNemar p = 0.134 and
p = 1.000). Its calibration barely moves.

**Train a specialist when something downstream reads the probability. Use a
hosted API with examples when only the argmax is consumed.**

## Quickstart

```bash
pip install -r requirements.txt
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

python src/gen_domain.py  --spec domains/banking.json --out data/banking
python src/train_domain.py --data-dir data/banking --adapter models/banking
python src/score_domain.py --adapter models/banking --data-dir data/banking \
    --out out/local_banking.jsonl
python src/serve.py --adapter models/banking
```

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "model":"local",
  "state":"My card got swallowed by the ATM yesterday. Also, when is my next payment due?",
  "questions":{"intent":{"type":"choice",
    "instructions":"What does this customer message ask for?",
    "criteria":{"damaged_card":"reporting a card that is physically damaged",
                "bill_due":"asking when a bill or payment is due",
                "out_of_scope":"not a banking or card request at all"}}}}'
```

```json
{"choice": "bill_due", "confidence": 0.455,
 "probabilities": {"bill_due": 0.564, "damaged_card": 0.336,
                   "out_of_scope": 0.000}}
```

## Adding a domain

A domain is a JSON spec, not a code change — label criteria, an abstain class,
confusable pairs, and where the labelled text comes from. See
[`domains/banking.json`](domains/banking.json) and
[GUIDE.md §3](GUIDE.md#3-step-1--write-the-domain-spec).

## Layout

| path | role |
|---|---|
| `domains/*.json` | domain specs: intents, criteria, abstain class, confusable pairs, data source |
| `src/gen_domain.py` | builds the corpus: in-domain, out-of-scope and genuinely ambiguous items |
| `src/train_domain.py` | LoRA + pointer head, cross-entropy to soft targets |
| `src/score_domain.py` | held-out scoring with real label names and per-kind breakdown |
| `src/ab_domain.py` | comparison against a hosted API, with matched precision and a few-shot arm |
| `src/domain_stats.py` | paired significance tests (McNemar, paired bootstrap) |
| `src/serve.py` | the HTTP server |
| `src/decision_api.py` | the request/response contract and three-level attention tree |
| `src/sym_options.py` / `src/pointer.py` | order-invariant option encoding and the pointer readout |
| `src/metrics.py` | proper scoring rules, adaptive ECE, risk–coverage, tie-aware accuracy |

## Design notes

- **Soft targets, not one-hot.** Cross-entropy to a soft target is uniquely
  minimised at `p = q`; for a single-step decision the policy gradient reduces
  to supervised learning, so no RL machinery is needed. Worth ΔKL −2.45 and
  Δr +1.02 against one-hot.
- **Option order randomised per item.** Worth +17.6 accuracy points.
- **Order-invariant option encoding.** Each option is an isolated branch over a
  shared prefix at identical positions, scored by a bilinear pointer head, so
  permuting options changes the output by ~1e-08. Options carry no label token,
  so K is unbounded.
- **Option subsampling in training only.** The pointer scores each option
  independently, so distractors can be dropped while training and the full set
  used at inference — 30.3 s/step to 2.4 s/step at K=31.
- **One forward pass per request.** Several questions about one state share a
  three-level attention tree, so Q questions cost one pass, not Q.

## Requirements

Python 3.10+, one GPU (8 GB is sufficient; see
[GUIDE.md §9](GUIDE.md#9-tuning-for-your-hardware) for other sizes). CPU works
but is considerably slower.

Comparing against a hosted API needs a key, supplied by `--key-file` or the
`DECISION_API_KEY` environment variable. Keys are never read from or written to
the repository.

## Licence

MIT — see [LICENSE](LICENSE). The reference domains are built from
[CLINC-150](https://huggingface.co/datasets/clinc/clinc_oos) (CC BY 3.0), which
is downloaded at build time and not redistributed here.
