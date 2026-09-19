# Training a domain decision model

A *decision model* takes a shared state and a typed question and returns a
calibrated probability distribution over options. No text is generated; the
answer is the distribution.

This guide produces one for a single domain: a LoRA adapter over a small
instruct backbone, trained on real labelled text, served behind a
`POST /v1/systemone` contract. Following it end to end takes about an hour of
compute on one consumer GPU.

Measured outcomes are in [RESULTS.md](RESULTS.md); the short version is in
[§10](#10-reference-results).

---

## 1. Decide whether you want this

Use a domain decision model when **something downstream reads the
probability** — a deferral threshold, an expected-cost rule, a confidence bar
below which a human is asked, a router that needs to know when two
destinations are equally right.

Use a hosted API with a few-shot prompt when **only the argmax is consumed**.
On the two domains measured here, a well-prompted hosted model matches or beats
a fine-tuned 1.5B on in-domain accuracy. The specialist's reliable advantage is
calibration, not discrimination.

Two prerequisites:

- **A closed set of decisions.** Write the domain in one sentence and list the
  labels. If the list is open-ended, this recipe does not apply.
- **Real labelled text in that domain.** Human-written utterances with human
  labels. This is where the domain knowledge comes from and there is no
  substitute for it.

---

## 2. Requirements

| | |
|---|---|
| Backbone | any small instruct model; `Qwen2.5-1.5B-Instruct` is the reference. Larger is better if it fits. |
| Accelerator | one GPU. See [§9](#9-tuning-for-your-hardware) for sizing; CPU works but is far slower. |
| Python | 3.10+, `pip install -r requirements.txt` |
| Data | a public or internal corpus of real labelled text (the reference uses CLINC-150) |

Set the backbone once:

```bash
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct    # or a local path
```

---

## 3. Step 1 — write the domain spec

Everything domain-specific is one JSON file. Adding a domain is a spec, not a
code change. See [`domains/banking.json`](domains/banking.json) and
[`domains/travel.json`](domains/travel.json).

```json
{
  "domain": "banking_support",
  "question": ["Which banking request is the customer making?",
               "What does this customer message ask for?",
               "Route this customer message to the right banking intent."],
  "source": {
    "train": {"kind": "hf", "dataset": "clinc/clinc_oos", "config": "plus",
              "split": "train", "text_field": "text", "label_field": "intent"},
    "test":  {"kind": "hf", "dataset": "clinc/clinc_oos", "config": "plus",
              "split": "test",  "text_field": "text", "label_field": "intent"}
  },
  "oos_label": "oos",
  "abstain_label": "out_of_scope",
  "abstain_criteria": "the message is not a banking or card request at all",
  "intents": {
    "balance": "asking the current balance of an account",
    "bill_due": "asking when a bill or payment is due"
  },
  "confusable": [["balance", "bill_balance"], ["bill_due", "pay_bill"]],
  "joiners": ["{a} Also, {b}", "Two things: {a} {b}"],
  "hedges":  ["{a} Although maybe what I really mean is: {b} Not certain which."]
}
```

Write each field to these rules:

**`intents`** — map each label to a criteria sentence. The criteria text is
what the model reads: options are rendered label-free, so the label string
never reaches the decision. Write *"asking when a bill or payment is due"*,
not *"bill due"*.

**`question`** — supply at least three paraphrases. One is sampled per item,
which keeps the model from keying on a fixed question string.

**`abstain_label` / `abstain_criteria`** — always include an abstain class. A
decision model that cannot say "not my problem" is not deployable, and this is
the only thing that teaches it.

**`confusable`** — pairs a person could plausibly conflate, or that could
appear together in one message. These generate the ambiguous items. Ten to
fourteen pairs is enough.

**`joiners` / `hedges`** — templates that combine two real utterances. Keep
each list the same length across spec variants if you plan to run the
paraphrase check in [§7](#7-step-5--validate-before-you-compare); the seeded
RNG then selects the same utterances and only the connective changes.

**`source`** — `kind: "hf"` loads from the Hugging Face hub; `kind: "parquet"`
with a `path` reads a local file. For any other corpus, add a branch to
`load_rows` in [`src/gen_domain.py`](src/gen_domain.py) that returns
`[(text, label), ...]`. Nothing else changes.

---

## 4. Step 2 — build the corpus

```bash
python src/gen_domain.py --spec domains/banking.json --out data/banking
```

The generator produces three kinds of item. Include all three.

### 4.1 In-domain, one-hot

Real utterances with their real labels, balanced across intents. Use text
written by people who wanted something; the phrasing variety is what
transfers.

### 4.2 Out-of-scope, one-hot

Messages that should be refused, from two sources: the corpus's own
out-of-scope class, and in-scope intents belonging to *other* domains, which
are genuine negatives you already have. A quarter of the hard items is a good
default.

### 4.3 Genuinely ambiguous, soft targets

Target roughly **45% soft items**. This fraction sets the uncertainty
correlation `r(H_true, H_pred)`: at 23% soft it measures +0.14, at 45–47% it
measures +0.23 to +0.33 on a general corpus and +0.93 to +0.95 in-domain.

Two constructions, both built from real text:

- **composed** (75% of soft items) — join two real utterances from a
  `confusable` pair. *"Could you share my credit score? Two things: what
  preventative measures can I take to avoid a low credit score"* →
  `{credit_score: 0.5, improve_credit_score: 0.5}`. Both intents are literally
  present.
- **hedged** (25%) — the sender **names both intents and says they are unsure
  which**, leaning to the one stated first → `{a: 0.67, b: 0.33}`.

> **The rule that governs every soft target:** each unit of probability mass
> must correspond to something a reader can underline in the text. If you
> cannot point at the words that justify a 0.33, the target is wrong and the
> model will learn to hedge where it should not.

### 4.4 Before training, read twenty items

```bash
head -20 data/banking/train.jsonl | python -m json.tool --json-lines
```

Confirm the text reads naturally, the targets match what the text says, and the
option order differs between items. The generator also enforces two invariants
and will tell you about them: the soft count is derived from the hard items
*actually* built (corpora cap utterances per intent), and train/test verbatim
overlap must be zero.

---

## 5. Step 3 — train

```bash
python src/train_domain.py \
  --data-dir data/banking --adapter models/banking \
  --epochs 2 --bs 4 --accum 4 --max-options 10
```

Measure step time first on any new hardware or domain:

```bash
python src/train_domain.py --data-dir data/banking --adapter /tmp/probe --probe
```

Four design points, and why they are set that way:

**Cross-entropy to the soft target.** For a single-step decision this is the
proper scoring rule you want: it is uniquely minimised at `p = q`, and the
policy gradient reduces to supervised learning, so no RL machinery is needed.
Soft targets rather than one-hot are worth ΔKL −2.45 and Δr +1.02 — the largest
single effect measured in this work.

**Randomised option order**, done in the generator. Worth +17.6 accuracy
points, the largest accuracy effect measured.

**Pointer readout over an order-invariant layout.** Each option is an isolated
branch over a shared `[state + question]` prefix, all branches at identical
positions, scored by a bilinear head:

```
z_i = <W_q h_dec, W_k h_opt_i> / sqrt(d)
```

Permuting options changes the output by ~1e-08, so order invariance is exact
rather than learned, and options need no single-token label, so K is
unbounded.

**Option subsampling during training** (`--max-options`). Long option lists
dominate step time, because an explicit 4D mask forces the attention path that
materialises a T×T matrix per head per layer. Keeping every option that carries
target mass plus a random sample of the rest cuts step time several-fold — on
the reference run, from 30.3 s/step to 2.4 s/step at K=31. This is sound
because the pointer scores each option independently from its own
representation, so nothing learned depends on how many options were present.

Confirm in the log that subsampling applies to training only:

```
data/banking/train.jsonl:   n=8455  (option-subsampled 8455)  ...
data/banking/heldout.jsonl: n=1173  (option-subsampled 0)     ...
```

---

## 6. Step 4 — score

```bash
python src/score_domain.py --adapter models/banking \
  --data-dir data/banking --out out/local_banking.jsonl
```

Report all of these. Accuracy alone will not tell you whether the model works.

| metric | what it measures |
|---|---|
| accuracy | discrimination |
| **tie-aware accuracy** | with 50/50 targets a plain argmax scores a coin flip as an error; ~14% of soft items have tied maxima |
| **KL(true‖pred)** | the training objective, on held-out data |
| **r(H_true, H_pred)** | whether the model is uncertain on the *right items* — the property you trained for |
| adaptive ECE | confidence against correctness, equal-mass bins |
| **abstention precision / recall / F1** | a model can score well overall and never abstain |
| **per-kind breakdown** | in-domain routing, abstention and ambiguity are three different jobs |

Report AURC alongside KL and `r`, never instead of them: it scores ranking, not
magnitude, and is insensitive to a model that is confidently wrong in a
consistent order.

---

## 7. Step 5 — validate before you compare

Two checks. Both are cheap and both change what you are entitled to claim.

### 7.1 Confirm the model learned the task, not your templates

Build a second held-out split that is item-by-item aligned with the first —
same utterances, same targets — but with different question paraphrases,
joiners and hedges. [`domains/banking_transfer.json`](domains/banking_transfer.json)
is such a spec. Because every template list keeps its original length, the
seeded RNG selects identical utterances and only the connective text differs.

```bash
python src/gen_domain.py --spec domains/banking_transfer.json --out data/banking_xfer
python src/score_domain.py --adapter models/banking \
  --data-dir data/banking_xfer --out out/local_banking_xfer.jsonl
```

Scores should hold. The reference model scores 0.836 / KL 0.176 on the shifted
split against 0.821 / 0.168 on the original.

### 7.2 Give the baseline a fair prompt

If you compare against a hosted API, put one labelled example per intent into
its `state` first. This is free for the API and it is the arm that decides
whether an accuracy claim survives.

```bash
python src/ab_domain.py --data-dir data/banking --fewshot 31 --limit 300 \
  --key-file ~/.decision_api_key
```

On the reference domains, examples moved the hosted model's determinate
accuracy +3.1 points (banking) and +1.8 (travel) — enough to erase the
specialist's entire apparent edge, while leaving its calibration nearly
unchanged.

### 7.3 Match precision when the baseline rounds its output

Hosted APIs commonly round probabilities. The reference API reports two
decimals: over 22,537 returned entries, 94.4% are exactly `0.0`, the smallest
non-zero value is exactly `0.01`, and rows sum to 0.99–1.00. Anything below
0.005 arrives as zero.

A KL computed against a rounded zero is determined by your epsilon, not by the
model: a target class holding 0.33 reported as `0.00` contributes
`0.33·log(0.33/ε)`, about 8 nats at ε=1e-12 and about 1.6 nats at ε=0.0025.

`ab_domain.py` therefore rounds **both** sides to the same precision, replaces
zeros with the midpoint of the interval they are known to occupy, and
renormalises. Report the sensitivity with `--floor-sweep`:

| floor | 0.005 | **0.0025** | 0.001 | 1e-4 | 1e-12 |
|---|---|---|---|---|---|
| hosted API, mean KL | 0.617 | **0.580** | 0.577 | 0.649 | 1.427 |
| local model, mean KL | 0.229 | **0.168** | 0.134 | 0.126 | 0.270 |

Quote the matched-precision figure and show the sweep.

### 7.4 Test the difference, not the two numbers

Both systems answer the same items, so the comparison is paired: McNemar for
accuracy, a paired bootstrap for KL.

```bash
python src/domain_stats.py --api out/api_banking.jsonl \
  --local out/local_banking.jsonl --tag BANKING
```

---

## 8. Step 6 — serve

```bash
python src/serve.py --adapter models/banking
```

`GET /` is a page for trying it by hand; `POST /v1/systemone` is the contract:

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "model":"local",
  "state":"My card got swallowed by the ATM yesterday. Also, can you tell me when my next payment is due?",
  "questions":{"intent":{"type":"choice",
    "instructions":"What does this customer message ask for?",
    "criteria":{"damaged_card":"reporting a card that is physically damaged",
                "bill_due":"asking when a bill or payment is due",
                "report_lost_card":"reporting a card lost or stolen",
                "out_of_scope":"not a banking or card request at all"}}}}'
```

```json
{"choice": "bill_due", "confidence": 0.455,
 "probabilities": {"bill_due": 0.564, "damaged_card": 0.336,
                   "report_lost_card": 0.100, "out_of_scope": 0.000}}
```

Several questions about one state are answered in a single forward pass over a
three-level attention tree (state → question branches → option sub-branches),
so Q questions cost one pass rather than Q. Question order and option order are
provably irrelevant. Supported question types are `choice`, `noul` and `score`.

---

## 9. Tuning for your hardware

Start from the reference settings and adjust to fit. The knobs are `--bs`,
`--accum` and `--max-options`; keep `bs × accum` constant to hold the effective
batch size at 16.

| available VRAM | suggested |
|---|---|
| 24 GB+ | a 3B backbone, `--bs 8 --accum 2 --max-options 16` |
| 12–16 GB | `--bs 8 --accum 2 --max-options 12` |
| 8–12 GB | `--bs 4 --accum 4 --max-options 10` *(reference)* |
| 6 GB | `--bs 2 --accum 8 --max-options 8` |
| CPU only | runs unchanged in fp32; expect roughly an order of magnitude longer |

Sizing notes:

- Cost scales with **sequence length**, which is driven by `--max-options` and
  by how long your criteria sentences are, more than by the number of items.
- `--max-options` never drops an option carrying target mass, so lowering it
  costs distractor variety, not correctness of the targets.
- Inference always uses the full option set regardless of this setting, so size
  the *scoring* pass for full K — it needs more memory per item than training
  does.
- Run one long job at a time. Two training processes on one accelerator will
  contend for memory and run far slower than either alone, and concurrent
  data-fetching jobs can starve the dataloader.

Reference run for scale: 1,056 optimiser steps over 8,455 items at K=31 took
2,580 s on a single 8 GB consumer GPU, plus about three minutes to score 1,173
held-out items.

---

## 10. Reference results

Two domains built with this pipeline and measured against a live hosted
decision model on 1,173 identical held-out items each, at matched precision.
Full detail and significance testing in [RESULTS.md](RESULTS.md).

| banking, K=31 | KL | acc | acc(tie) | r | ECE_ad | abstain F1 |
|---|---|---|---|---|---|---|
| hosted API, zero-shot | 0.580 | 0.790 | 0.939 | +0.343 | 0.091 | 0.940 |
| local 1.5B | **0.168** | **0.821** | **0.978** | **+0.933** | **0.089** | **0.962** |

| travel, K=25 | KL | acc | acc(tie) | r | ECE_ad | abstain F1 |
|---|---|---|---|---|---|---|
| hosted API, zero-shot | 0.573 | 0.772 | 0.943 | +0.428 | 0.103 | 0.960 |
| local 1.5B | **0.124** | **0.825** | **0.984** | **+0.950** | **0.091** | **0.970** |

With one labelled example per intent in the API's state (300 paired items):

| | acc determinate | KL | r |
|---|---|---|---|
| banking: API few-shot | **0.982** | 0.486 | +0.545 |
| banking: local | 0.957 | **0.197** | **+0.935** |
| travel: API few-shot | **0.976** | 0.444 | +0.531 |
| travel: local | 0.970 | **0.118** | **+0.948** |

Determinate-accuracy difference is not significant in either domain
(McNemar p = 0.134 and p = 1.000). The calibration difference is large and
significant in both.

---

## 11. Adapting this to your own corpus

1. Write `domains/<yours>.json`. Only `intents`, `confusable`, `question`,
   `abstain_*` and `source` are domain-specific.
2. If your corpus is not on the Hub and not a parquet file with a class-index
   label column, add a branch to `load_rows` returning `[(text, label), ...]`.
3. If your out-of-scope class has a different name, set `oos_label`.
4. If you have no natural out-of-domain negatives, build them from another
   corpus; abstention needs real negatives, not synthesised ones.
5. Everything else — training, scoring, the comparison harness, serving — is
   domain-independent.
