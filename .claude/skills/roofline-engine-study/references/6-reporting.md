# Phase 6 — Reporting

## Structure of a result section

1. One-line answer with scope ("on this device, model, batch 1, greedy, vs <baseline default path>").
2. Conditions and controls (what was fixed, randomized, gated, frozen; commit of the registered plan).
3. Decision rule in one sentence.
4. Table per scenario: descriptive ratio, paired median, per-session values, adjusted p, verdict.
5. Every void and undecided with its concrete reason and data.
6. Descriptive comparators.
7. What this cannot show.

## Evidence levels

| Level | Example wording |
|---|---|
| Pre-registered verdict | "faster under the registered rule (sign test on paired median > 1.02, Holm, both sessions)" |
| Descriptive measurement | "geometric-mean ratio 1.05; descriptive bootstrap interval [..]" |
| Micro-benchmark | "in isolation, the GEMV reached ~95 GB/s" — not a whole-engine claim |
| Hypothesis | "consistent with throttling; not separated causally" |
| Reference line | "~98% of the calibrated reference" — never "of the hardware limit" |

## Language to avoid

- "Proves optimal", "only 1–4% left" — the reference rates are not certified ceilings.
- "Equivalent" from an interval inside a tolerance; "no effect" from a non-significant test.
- "Free" for a precision or KV policy whose cost is only measured by KL on one corpus.
- Attributing gains to kernels when scheduling differs (use the fair-schedule baseline or say "not attributed").
- Cross-day or cross-device generalization from one night on one machine.

## Bookkeeping

- Keep raw records, original reports and the frozen analysis; corrections get new files and a note
  explaining when and why (`NOTE.md`, "post-collection" headers).
- Update every place that quotes the old numbers (main report, README, per-experiment README) in one commit,
  so the repository never shows two conflicting verdicts.
- Store the long-term conclusion in project memory if the agent has one.
