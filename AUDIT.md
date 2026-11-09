# Audit (the study report)

Self-audit of mlops-evidence-lab against its own frozen
protocol (PREREGISTRATION.md) and the standing project
standard. Written after the runs it reports on. Every number below
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
  not install it (DECISIONS.md D5). Under the portfolio
  standard v2 a 3.11 leg returns for the dependency-free checks
  only: layout, lock, no public grade, and results regeneration
  (DECISIONS.md D7).

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

## How the evidence is protected

1. Pre-registration before results. PREREGISTRATION.md is the
   repository's first commit (2f5de92), made before any code or
   result; git history shows the order, and the file has not changed
   since. tests/test_layout.py pins its SHA-256, so CI fails if one
   byte changes; amendments go to DECISIONS.md instead.
2. Verdicts are code, not prose. scripts/make_results.py computes
   every verdict from the frozen pass lines and the recorded
   artifacts; tests/test_units.py exercises every branch of the H1 to
   H4 rules. CI regenerates RESULTS.md and results/verdicts.json and
   fails if either differs from the committed file.
3. Pinned subjects and data. Both subject repositories are pinned to
   commits and never modified; the credit data file is SHA-256
   verified against the subject's own manifest before any run.
4. Refusals are recorded. Every Gemini request, retry and status is
   in results/gemini_call_log.jsonl, including the 429s that stopped
   Half 2. No other provider or key was used to route around them.
5. Grades are kept outside this repository; this report carries the
   evidence and the verdicts only.

## Threats to validity

- H3 and H4 not run. Half of the research question is unanswered in
  this delivery: the harness, samples, and scorer are complete and
  the blocker is external and documented, but the standardized
  evaluation itself, the second piece of evidence the study exists
  to produce, has no scores.
- H2's promote direction untested. The frozen design allowed this
  outcome and the verdict reports it honestly, but the gate's
  positive path has no evidence behind it.
- Post-freeze harness corrections: the skops allowlist, the scorer
  retry change, and the probe gating were all necessary and all
  recorded, but a cleaner freeze would have needed fewer of them.
- CI's full suite covers one interpreter (3.12) where the house
  pattern is two; the 3.11 leg runs only the dependency-free checks
  (DECISIONS.md D5, D7).
- No interval on any headline number. H1 is one deterministic run
  (seed 42, one fixed 70/30 split); it shows the published number is
  reproducible, not how much it moves across splits or seeds. The
  subject's own 5-seed mean (0.7830, results/rigor.json in
  credit-risk-lab) is the nearest spread estimate and was not rerun
  here.
- No ablations. The ML research profile asks for one-part-at-a-time
  removals; this study registered none, so the ML profile is not met
  even for Half 1.
- One machine of record (the Muse VM, htch-runtime). Numbers are
  claimed for that machine and the pinned library versions in
  results/half1_manifest.json.

## What completes the study

The route reopening. One command
(.venv/bin/python src/score_ragas.py) then the results
generator turns H3 and H4 from NOT RUN into measured verdicts,
publishes the full per-question failure table, and answers the
second half of the question. Nothing else in the repository
needs to change for that to happen, which is the point of
building the harness before the window closed.

Addendum (same day, after the section above was first written): two
final probes returned one 200 and then a 429 two minutes
later. The route trickles single calls; it does not sustain a
run. The NOT RUN verdicts are unchanged
(DECISIONS.md D6).
