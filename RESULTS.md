# Results

Two domain decision models trained with [GUIDE.md](GUIDE.md) and measured
against a live hosted decision model (TypeSafe's `jev-latest`) on identical
items.

The hosted API was used only as a benchmark. Neither model was trained on its
outputs; all labels come from CLINC-150's human annotations.

---

## 1. Setup

| | banking | travel |
|---|---|---|
| intents + abstain (K) | 31 | 25 |
| training items | 8,455 | 7,364 |
| held-out items | 1,173 | 1,173 |
| soft-target fraction | 45.0% | 45.0% |
| train/test verbatim overlap | 0 | 0 |
| optimiser steps | 1,056 | 920 |

Backbone `Qwen2.5-1.5B-Instruct`, LoRA r=16 α=32 on all projection modules,
pointer readout over an order-invariant option layout, cross-entropy to soft
targets, 2 epochs, effective batch 16, `--max-options 10`. Single 8 GB
consumer GPU.

Held-out items are drawn from CLINC's test split, so the utterances are unseen.
Item kinds: `in_domain` 480, `abstain_oos` 82, `abstain_other_domain` 83,
`ambiguous` 528.

---

## 2. Measurement: matched precision

The hosted API rounds probabilities to two decimals. Over 22,537 returned
entries, 94.4% are exactly `0.0`, the smallest non-zero value is exactly
`0.01`, and each row sums to between 0.99 and 1.00. Every probability below
0.005 arrives as zero.

A KL against a rounded zero is therefore determined by the epsilon chosen, not
by the model. A target class holding 0.33 reported as `0.00` contributes
`0.33·log(0.33/ε)`: about 8 nats at ε=1e-12, about 1.6 nats at ε=0.0025.

All figures below round **both** systems to two decimals, replace zeros with
0.0025 — the midpoint of the interval `[0, 0.005)` a reported zero is known to
lie in — and renormalise. Mean KL by choice of floor, banking, n=1,173:

| floor | 0.005 | **0.0025** | 0.001 | 1e-4 | 1e-12 |
|---|---|---|---|---|---|
| hosted API | 0.617 | **0.580** | 0.577 | 0.649 | 1.427 |
| local | 0.229 | **0.168** | 0.134 | 0.126 | 0.270 |

The ordering is stable across the whole range; only the unguarded floor
inflates both.

A related detail: the API's `confidence` field is `(p_max − 1/K)/(1 − 1/K)`
computed on the unrounded `p_max`, so inverting it recovers `p_max` more
precisely than the `probabilities` field does.

---

## 3. Both domains, hosted API zero-shot

| banking, K=31 | KL | acc | acc(tie) | r(H_t,H_p) | ECE_ad | AURC | abstain F1 |
|---|---|---|---|---|---|---|---|
| jev-latest | 0.5800 | 0.7903 | 0.9386 | +0.3431 | 0.0905 | 0.0981 | 0.9398 |
| local 1.5B | **0.1683** | **0.8210** | **0.9778** | **+0.9325** | **0.0890** | **0.0563** | **0.9617** |

| travel, K=25 | KL | acc | acc(tie) | r(H_t,H_p) | ECE_ad | AURC | abstain F1 |
|---|---|---|---|---|---|---|---|
| jev-latest | 0.5731 | 0.7724 | 0.9429 | +0.4277 | 0.1030 | 0.0999 | 0.9598 |
| local 1.5B | **0.1237** | **0.8252** | **0.9838** | **+0.9503** | **0.0912** | **0.0465** | **0.9699** |

Accuracy by item kind (API → local):

| kind | n | banking | travel |
|---|---|---|---|
| in_domain | 480 | 0.938 → 0.958 | 0.935 → 0.973 |
| abstain_oos | 82 | 1.000 → 0.976 | 0.927 → 0.963 |
| abstain_other_domain | 83 | 0.988 → 1.000 | 0.952 → 0.988 |
| ambiguous | 528 | 0.593 → 0.644 | 0.572 → 0.644 |
| *ambiguous, mean KL* | | *0.995 → 0.117* | *0.972 → 0.089* |

### Paired tests

McNemar on accuracy, paired bootstrap on KL:

| | local | api | diff [95% CI] | McNemar |
|---|---|---|---|---|
| banking, acc all | 0.8210 | 0.7903 | +0.031 [+0.006, +0.056] | p = 0.021 |
| banking, acc determinate | 0.9659 | 0.9519 | +0.014 [−0.003, +0.031] | p = 0.164 (n.s.) |
| banking, acc ambiguous | 0.6439 | 0.5928 | +0.051 [−0.002, +0.102] | p = 0.064 (n.s.) |
| banking, KL | 0.1683 | 0.5800 | −0.412 [−0.455, −0.368] | — |
| travel, acc all | 0.8252 | 0.7724 | +0.053 [+0.028, +0.078] | p < 0.001 |
| travel, acc determinate | 0.9736 | 0.9364 | +0.037 [+0.019, +0.056] | p < 0.001 |
| travel, KL | 0.1237 | 0.5731 | −0.449 [−0.496, −0.403] | — |

On banking the determinate-accuracy difference is not significant. On travel
it is — but see §5, which shows that result depends on the baseline being
zero-shot.

### Error structure

The dominant error for both systems is over-abstention: routing an in-domain
message to `out_of_scope`. On banking in-domain items the API makes 30 errors
in 480, the local model 20. Abstention precision/recall: local 0.937/0.988,
API 0.891/0.994.

On ambiguous items the API is not mis-routing. Its argmax falls inside the true
support on 98–100% of items, and its tie-aware accuracy is 0.939. It assigns
little mass to the second intent:

| banking items | argmax in support | mean p on weaker target | mean p_max |
|---|---|---|---|
| composed 50/50 (n=400) | 98.0% | 0.111 | 0.877 |
| hedged 67/33 (n=128) | 100.0% | 0.129 | 0.863 |
| determinate (n=645) | 95.2% correct | — | 0.955 |

Mean confidence does fall on ambiguous items, from 0.955 to 0.877. The targets
call for 0.5. That difference accounts for the calibration gap.

---

## 4. Template dependence

A paraphrase-shifted banking split: item-by-item aligned with the original
(same targets, same kinds, same underlying utterances), with three new question
paraphrases, four new joiners and three new hedge templates. The 645
determinate items have byte-identical text; only the question changed.

The shifted split is not intrinsically harder — the API scores the same on
both:

| jev-latest on | acc | acc det | acc amb | KL |
|---|---|---|---|---|
| original templates | 0.7903 | 0.9519 | 0.5928 | 0.5800 |
| shifted templates | 0.7945 | 0.9550 | 0.5985 | 0.4972 |

McNemar b=28, c=23, p=0.575.

The local model holds up, scoring marginally higher on the shifted split:

| local banking model on | acc | acc det | KL |
|---|---|---|---|
| original templates | 0.8210 | 0.9659 | 0.1683 |
| shifted templates | 0.8363 | 0.9643 | 0.1755 |

No template dependence.

---

## 5. The hosted API with a fair prompt

The `state` field accepts free text, so the API can be given the domain's label
set by example: one training-split utterance per intent, 31 for banking and 25
for travel. No training, only a longer prompt. 300 paired items:

| banking | acc | acc determinate | acc ambiguous | KL | r |
|---|---|---|---|---|---|
| jev zero-shot | 0.7933 | 0.9512 | 0.6029 | 0.5888 | +0.3299 |
| jev 31-shot state | 0.8067 | **0.9817** | 0.5956 | 0.4858 | +0.5453 |
| local 1.5B | 0.8133 | 0.9573 | 0.6397 | **0.1968** | **+0.9352** |

| travel | acc | acc determinate | acc ambiguous | KL | r |
|---|---|---|---|---|---|
| jev zero-shot | 0.8167 | 0.9581 | 0.6391 | 0.5786 | +0.4223 |
| jev 25-shot state | 0.8300 | **0.9760** | 0.6466 | 0.4439 | +0.5311 |
| local 1.5B | 0.8333 | 0.9701 | 0.6617 | **0.1176** | **+0.9484** |

Determinate-accuracy McNemar, local vs few-shot API: banking b=0, c=4,
p=0.134; travel b=2, c=3, p=1.000. **No significant difference in either
domain, and on banking the point estimate favours the API.**

Examples are worth +3.1 determinate-accuracy points on banking and +1.8 on
travel. That is enough to remove the specialist's advantage in §3, including
the significant travel result.

Examples do not transfer calibration. The API's `r` rises from +0.330 to +0.545
(banking) and +0.422 to +0.531 (travel), roughly half the local model's, and
its KL stays 2.5–3.8× higher.

This arm uses 300 items against 1,173 in §3, so its intervals are wider. The
direction is consistent across two independent domains.

---

## 6. Serving

A live request against the served banking adapter:

> *"My card got swallowed by the ATM yesterday. Also, can you tell me when my
> next payment is due?"*

```
choice: bill_due   confidence 0.455
  bill_due          0.564
  damaged_card      0.336
  report_lost_card  0.100
  out_of_scope      0.000
round trip: 540 ms
```

Mass is divided between the two intents present and confidence is low
accordingly, on a 1.5B model on one 8 GB consumer GPU.

---

## 7. Conclusions

1. **The specialist's advantage is calibration.** In-domain it reaches
   `r(H_true, H_pred)` of 0.93–0.95 and KL of 0.12–0.17, against a frontier
   decision model's 0.53–0.55 and 0.44–0.49 when that model is fairly prompted.
2. **It is not an accuracy advantage.** Against a few-shot baseline the
   determinate-accuracy difference is not significant in either domain.
3. **Choose on what consumes the output.** Argmax routing: use the hosted API
   with examples. Threshold, deferral or expected-cost logic that reads the
   probability: the specialist is markedly better calibrated, runs locally at
   roughly half a second per request, and has no per-call cost.
4. **The recipe transfers.** Two domains with different K, different confusable
   structure and different label semantics, one unchanged pipeline, the same
   qualitative outcome.

### Limitations

- Both domains come from CLINC-150. The construction of ambiguous items is
  shared between training and evaluation, though the utterances are not; a
  corpus with naturally occurring multi-intent messages would test this better.
- The few-shot arm covers 300 items per domain, not 1,173.
- One backbone at one size. The effect of scale on the calibration gap is not
  measured here.
- Soft targets of 0.5/0.5 for a two-intent message, and 0.67/0.33 for a stated
  lean, are defensible readings rather than ground truth. Tie-aware accuracy is
  reported alongside so a reader who prefers "either answer is correct" can
  apply that standard instead.

---

## 8. Reproduce

```bash
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
python src/gen_domain.py --spec domains/banking.json --out data/banking
python src/train_domain.py --data-dir data/banking --adapter models/banking \
  --epochs 2 --bs 4 --accum 4 --max-options 10
python src/score_domain.py --adapter models/banking --data-dir data/banking \
  --out out/local_banking.jsonl
python src/ab_domain.py --data-dir data/banking --key-file ~/.decision_api_key \
  --floor-sweep
python src/ab_domain.py --data-dir data/banking --fewshot 31 --limit 300 \
  --key-file ~/.decision_api_key
python src/domain_stats.py --api out/api_banking.jsonl \
  --local out/local_banking.jsonl --tag BANKING
```

Swap `domains/banking.json` for `domains/travel.json` and `data/banking` for
`data/travel` to reproduce the second domain.
