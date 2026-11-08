# Audit

Self-audit of mlops-evidence-lab against its own frozen
protocol (PREREGISTRATION.md) and the standing project
standard. Written after the runs it grades. Every number below
is in RESULTS.md or in a results/ artifact.

## What the evidence supports

- Half 1 is complete and exact. Four MLflow runs on the frozen
  split reproduce the subject's published figures to the last
  printed digit on both models the subject publishes numbers
  for (champion AUC 0.7810, KS 0.4196, Brier 0.1355; baseline
  AUC 0.7518, KS 0.4008, Brier 0.1416). The registry holds the
  champion as credit_pd_champion version 1 under the champion
  alias, verified by a client read after the runs.
- The gate ran once per candidate and blocked both, with
  per-criterion reasons recorded. The negative control behaved
  as a control should (AUC 0.4828, near chance).
- H2 is INCONCLUSIVE by the frozen logic, not by editorial
  choice: the challenger missed the frozen bar, so the promote
  direction was never exercised. The challenger was not
  re-tuned after the fact; doing so would have broken the
  registration to manufacture a cleaner headline.
- Half 2's local half is measured and published: the frozen
  40-question set, the pipeline's answers, and the retrieval
  record (results/rag_samples.json), including the 10 of 12
  adversarial questions the pipeline answers when it should
  decline.

## Incidents and corrections, on the record

- The MLflow model logging needed an explicit skops
  trusted-types allowlist before the pipeline artifact would
  serialize (MLflow 3.18). The failed partial stores were
  deleted and the recorded runs come from one clean store.
- The first scoring attempt (2026-11-08) met HTTP 429 on every
  generateContent call through the full registered backoff
  ladder. The run was stopped at 0 of 160 completed metric
  jobs, so no partial scores exist. Spaced single-attempt
  probes over the following hour also returned 429 (the call
  log holds 3 smoke calls from wiring verification, 16 failed
  attempts from the stopped run, and the probes). Scoring was
  not routed around the refusal: the protocol names one judge
  route, and the refusal is reported as a result about that
  route's capacity, consistent with the sibling study
  (document-intelligence-lab recorded 27 of 89 calls rejected
  the same day). DECISIONS.md D4 records the restart gating;
  the Ragas retry count inside the harness was set from 2 to 1
  for the restart so a closed window burns fewer calls per job.
- CI runs Python 3.12 only: the pinned stack (numpy 2.5.x)
  requires 3.12 or newer, so the sibling-style 3.11 leg could
  not install it (DECISIONS.md D5).

## Jev claim check

The headline verdicts (H1 CONFIRMED, H2 INCONCLUSIVE, H3 and
H4 NOT RUN with the reason) were checked with the Jev
claim_check gate: model jev-1.13.0, noul 0.58, route REVIEW
(house bands: clear at 0.85 or above, block at 0.10 or below).
Spec and raw response: results/jev-claim-spec.json and
results/jev-claim-check.json. On a REVIEW route the figures
were hand-verified against the artifacts: rerun AUC, KS and
Brier against results/half1_metrics.json; challenger and
control AUCs against the same file; the frozen bar arithmetic
(0.7810 + 0.003 = 0.7840) against results/gate_decisions.json;
the 429 account against results/gemini_call_log.jsonl. Every
figure in the claim matches its artifact. Jev gates claims; it
authorizes nothing, and nothing here needed authorizing.

## Grade: 5,750 of 10,000

Deductions, itemized:

- H3 and H4 not run. Half of the research question is
  unanswered in this delivery: the harness, samples, and
  scorer are complete and the blocker is external and
  documented, but the standardized evaluation itself, the
  second piece of evidence the study exists to produce, has
  no scores. Minus 3,500.
- H2's promote direction untested. The frozen design allowed
  this outcome and the verdict reports it honestly, but the
  gate's positive path has no evidence behind it. Minus 500.
- Post-freeze harness corrections: the skops allowlist, the
  scorer retry change, and the probe gating were all necessary
  and all recorded, but a cleaner freeze would have needed
  fewer of them. Minus 150.
- CI covers one interpreter (3.12) where the house pattern is
  two. Minus 100.

What would raise the grade: the route reopening. One command
(.venv/bin/python src/score_ragas.py) then the results
generator turns H3 and H4 from NOT RUN into measured verdicts,
publishes the full per-question failure table, and removes
most of the first deduction. Nothing else in the repository
needs to change for that to happen, which is the point of
building the harness before the window closed.

Addendum (same day, after the grade above was written): two
final probes returned one 200 and then a 429 two minutes
later. The route trickles single calls; it does not sustain a
run. The grade and the NOT RUN verdicts are unchanged
(DECISIONS.md D6).
