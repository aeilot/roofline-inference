# Registered verification plan — <device> / <model> / <date>

Status: registered before formal collection. Commit: <fill after commit>. Changes after the first formal
measurement go to a separate post-collection file; this file stays as is.

## Question

Is <candidate engine> faster than <baseline engine, default path> on <device>, <model, dtype>, batch <n>,
<greedy>, for the scenarios below? This is not a quality non-inferiority test, not a throughput/serving
test, and does not generalize to other hardware.

## Engines and environment

| Id | Engine | Version / commit | Non-default settings |
|---|---|---|---|
| E0 | <baseline> | | |
| E2 | <candidate> | | |
| V0 | <descriptive comparator, optional> | | |

Shared interpreter/environment: <path, versions>. Model snapshot: <id>; weight SHA-256 recorded per session.

## Scenarios

| Id | Prompt tokens | Generated tokens | Metrics |
|---|---|---|---|
| ttft16 | 16 | 8 | TTFT |
| ... | | | |

Pairs per scenario: <N> (session 1: <N/2>, session 2: <N/2>), prompt kinds alternating random / text
(corpus snapshot <hash>). Sample size justified by pilot log-ratio SD: <values>.

## Procedure

- One fresh process per timed request; one engine loaded; full-length warm-up with different content.
- Gate before each request: thermal <rule>, AC power, then <idle s / warm workload>.
- Engine order within a pair and scenario order within a block randomized, seed <seed>.
- Timing: TTFT at first token; decode = median inter-token gap excluding the first two; request = until all
  engine work is synchronized. Fixed generated length, EOS ignored. Load/warm-up/cooling excluded.
- Two sessions, gap <h>; refuse to continue if code, environment, weights or corpus hashes differ.

## Exclusion and retry

Exclude a timed request if: gate timeout, worker crash, not on AC before/after, or the system slept during
the measurement. Each excluded unit is re-run once at the end of its session; the first attempt is kept
apart and not analysed. A scenario missing planned pairs is "insufficient".

## Correctness

- FP32 numerical screen for E0 and E2 at session start (criteria: <list>); stop if either fails.
- Generated-sequence divergences: FP32 forward at the first divergence; void the scenario only if the
  candidate's token is FP32-worse than E0's by more than <threshold, calibrated at the tested context lengths>.
- Comparators without logits: token screen on <fixtures>.

## Decision rule

Per scenario: one-sided exact sign test that the median of t_E0 / t_E2 exceeds <1.02>; Holm within families
<TTFT / decode / request>. "Faster" requires adjusted p < 0.05, both sessions present, each session's
estimate > 1. Otherwise "undecided"; correctness failures give "void" with the reason. Geometric means and
bootstrap intervals are descriptive. V0 comparisons are descriptive only.

## Known limitations (state in the report)

<same-night sessions? attribution not measured? context lengths without logit checks? ...>
