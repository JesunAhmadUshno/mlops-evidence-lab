# Decisions

## D1: the rerun re-implements the subject pipeline

Half 1 re-implements credit-risk-lab's champion training in
src/credit_common.py from its published source at the pinned
commit, changing only the tracking layer. The alternative,
importing the subject repo as a package, would couple this lab
to a repo whose own runs write into its results directory; the
rerun must not touch the subject. The re-implementation is
line-faithful for loading, feature construction, preprocessing,
split, and model configurations, and H1 is exactly the test of
whether that faithfulness reproduces the published number.

## D2: ragas 0.4.3 needs a one-module compatibility shim

ragas 0.4.3 imports ChatVertexAI from
langchain_community.chat_models.vertexai, which
langchain-community 0.4 removed (the class moved to
langchain-google-vertexai). Downgrading the langchain stack to
the 0.3 line conflicts with ragas 0.4.3's own requirements.
src/ragas_compat.py registers the class from its new home
under the old module path before ragas is imported. ragas uses
the class only in an isinstance dispatch list; no Vertex AI
route is used anywhere in this lab. Pinned versions are in
requirements.txt.

## D3: the question set's paraphrase share

The subject's deep set has 10 paraphrase questions; Half 2
takes 4 of them (p01, p02, p05, p06) beside all 24 regression
questions, keeping the in-corpus slice at 28 so the adversarial
slice is exactly 30 percent of the 40-question set, as
registered. The 4 chosen paraphrases cover four distinct
expected documents (water-cycle, volcanoes, northlight-events,
photosynthesis).

## D4: the first scoring attempt met a closed rate-limit window

The first Half 2 scoring run (2026-11-08, started 13:57 UTC)
met HTTP 429 on every generateContent attempt, including
attempts after the full registered backoff ladder; 16 failed
attempts are in results/gemini_call_log.jsonl. The run was
stopped with zero of 160 metric jobs completed, so no partial
result exists and nothing is contaminated. Scoring restarts
from zero when the route answers again, under the same frozen
protocol (the registration freezes a minimum 4.0 s spacing, so
slower pacing remains inside it if needed). Two harness
parameters, not protocol terms, change for the restart: the
Ragas RunConfig retry count goes from 2 to 1 so a closed
window burns fewer quota calls per job, and the run's start is
gated on two spaced probe calls succeeding. The route's
availability is part of what this half measures; the incident
is reported in AUDIT.md, not hidden.

## D5: CI runs Python 3.12 only

The first CI matrix (3.11 and 3.12, the sibling pattern)
failed on 3.11 because the pinned numerical stack requires
Python 3.12 or newer (numpy 2.5.x declares Requires-Python
>= 3.12). CI now runs the machine-of-record interpreter, 3.12,
only. The deduction is recorded in AUDIT.md.

## D6: after the audit was written, the route trickled, not opened

Two final probes after the results commit: the first returned
200, the second, two minutes later, returned 429. The route
is releasing roughly isolated calls, not a usable window; a
scoring run needs several hundred calls. The D4 gate (two
spaced successes) therefore still holds scoring back, and the
NOT RUN verdicts stand as delivered. When the route sustains
two spaced probes, the completion is the scorer command plus
the results generator, with no other change needed.

## D7: challenger-ensemble registered by amendment (Owner's tap, 2026-11-17)

H2 is INCONCLUSIVE because no registered candidate cleared the
frozen bar, so the gate's promote direction is untested. The
Owner approved one additional candidate on 2026-11-17, proposed
in docs/challenger-proposal-2026-11-17.md. This amendment
registers it before any run:

- Candidate name: challenger-ensemble. An equal-weight (1/3
  each, fixed, never fitted) soft-voting average of three
  pipelines, each with the frozen preprocessing of
  PREREGISTRATION.md, each fit on the frozen split
  (load_split(), stratified 70/30, random_state 42):
  (A) HistGradientBoostingClassifier(max_iter 300,
  learning_rate 0.06, max_depth 3, random_state 42);
  (B) XGBClassifier(n_estimators 200, learning_rate 0.04,
  max_depth 3, min_child_weight 10, subsample 0.8,
  colsample_bytree 0.8, reg_lambda 0.1, random_state 42);
  (C) LGBMClassifier(n_estimators 200, learning_rate 0.04,
  num_leaves 15, min_child_samples 40, subsample 1.0,
  subsample_freq 1, colsample_bytree 0.8, reg_lambda 1.0,
  random_state 42).
- Dependencies added to requirements.txt at the sibling study's
  pins: xgboost==3.4.1, lightgbm==4.7.0. No other dependency
  changes.
- Training runs once, as one MLflow run named
  challenger-ensemble in the existing experiment
  credit-risk-lab-model-line, from a new script
  src/train_challenger_ensemble.py. The frozen four candidates
  are not re-run (the gate asserts exactly one run per name).
  Metrics are computed by the existing metrics_for on the
  frozen test split and logged with the full parameter set,
  as in src/train_candidates.py.
- No configuration change, re-weighting, or member swap after
  the run's metrics exist. One candidate, one run.

## D7 result: challenger-ensemble blocked; H2 remains INCONCLUSIVE

Executed 2026-11-17 under the D7 amendment above. Training ran
once, as registered: src/train_challenger_ensemble.py produced
the single MLflow run challenger-ensemble
(584745882e9c44399e8b98ecde7f8de0) in experiment
credit-risk-lab-model-line. On the frozen test split, against
the champion-rerun run (AUC 0.7810, KS 0.4196, Brier 0.1355):

- challenger-ensemble: AUC 0.7825, KS 0.4235, Gini 0.5650,
  Brier 0.1350, score PSI 0.0013. Calibration table in
  results/calibration_challenger-ensemble.json.
- The frozen gate (.venv/bin/python src/gate.py
  challenger-ensemble) returned block: AUC 0.7825 against the
  required 0.7840 (champion + 0.003), fail; Brier 0.1350
  against the cap 0.1375 (champion + 0.002), pass. The
  decision, with the metric table and per-criterion reasons,
  is appended to results/gate_decisions.json. Nothing was
  registered; the alias champion stays on version 1.
- Verdict, by the mapping fixed in the proposal before the
  run: H2 remains INCONCLUSIVE. The ensemble beat the champion
  on all three headline metrics (+0.0015 AUC, Brier 0.1350
  against 0.1355, KS tied for the field best) but not by the
  frozen margin, so the gate's promote direction remains
  untested with a qualifying candidate. RESULTS.md was
  regenerated by scripts/make_results.py (verdicts unchanged:
  H1 CONFIRMED, H2 INCONCLUSIVE, H3 and H4 NOT RUN); the
  README headline is unchanged because the verdict did not
  change. Per D7, no further candidate follows from this
  amendment.

Two implementation notes, neither a protocol term. The model
artifact is logged with cloudpickle serialization: skops,
used for the pure-sklearn candidates, refuses the XGBoost
and LightGBM boosters even with trusted types named
(pre-run validation record). Before the registered run, the
soft-voting implementation was verified to equal the plain
arithmetic mean of the three members' probabilities exactly
(max difference 0.0 on the test split). The registered
dependency pins installed exactly (xgboost 3.4.1,
lightgbm 4.7.0); the GPU-only nvidia-nccl wheel failed to
download twice on network errors and was not installed,
because XGBoost needs it only on its GPU collective path,
which this study does not use; numpy and scipy were already
present at compatible versions.

## D8: bounded resolve attempt on the judge route; H3 and H4 remain NOT RUN (2026-11-17)

Under Jesun's resolve-everything order of 2026-11-17, one
bounded attempt was made on the D4/D6 block, exactly inside the
registered protocol. No route was switched, no other model or
key was used, and nothing was paid (FREE-ONLY).

- Probes (scripts/probe_route.py, the registered gating tool,
  single attempt each): the first returned HTTP 200; the
  second, spaced two minutes later on the D6 precedent,
  returned HTTP 200. The D4 restart gate (two spaced probe
  successes) was therefore met, and the registered scoring
  command (.venv/bin/python src/score_ragas.py) was started
  once, at about 22:03 EST.
- The route did not sustain the run. The attempts logged to
  results/gemini_call_log.jsonl between 2026-11-18T03:03:18Z
  and 03:06:32Z are 19 generate attempts: 17 HTTP 429 and 2
  HTTP 200. The failed attempts passed the D4 incident count
  (16) with only isolated successes between them, the D6
  pattern of a route releasing isolated calls rather than a
  usable window. The run was stopped at that point, per the
  registered threshold.
- The run completed zero of 160 metric jobs, so no partial
  result exists and nothing is contaminated: no
  ragas_per_question.csv and no ragas_summary.json were
  written, RESULTS.md and results/verdicts.json are untouched,
  and scoring still restarts from zero when the route sustains
  a window. H3 and H4 remain NOT RUN. The completion, when the
  route sustains two spaced probes and then the run itself,
  remains the scorer command plus the results generator, with
  no other change needed.

## D9: the judge instrument is switched to a local model by amendment (2026-11-19)

Part 1: the third registered Gemini attempt (2026-11-19, under
Jesun's finish-everything order).

- Probes (scripts/probe_route.py, single attempt each): HTTP
  200 at 08:04:42Z, 08:06:14Z, and 08:08:22Z. The 08:06:14 and
  08:08:22 pair is spaced 128 seconds, so the D4 restart gate
  was met on the D8 precedent. Tests before the run: 11 passed.
- The registered scoring command (.venv/bin/python
  src/score_ragas.py) was started once, at about 08:09 UTC.
  The route degraded inside the run: 142 attempts logged to
  results/gemini_call_log.jsonl between 08:09:22Z and
  08:28:09Z (47 HTTP 200, 93 HTTP 429, 2 timeouts). The last
  200 was at 08:22:14Z; the 36 attempts after it were all 429.
  Ragas internal progress reached 25 of 160 jobs. The run was
  stopped at 08:28 UTC on the D6/D8 signature: a route
  releasing isolated calls, then none.
- The scorer writes outputs only on completion, so no
  ragas_per_question.csv and no ragas_summary.json were
  written; nothing is contaminated. The Gemini-instrument
  state is unchanged: H3 and H4 remain NOT RUN under the
  Gemini instrument.
- Across three registered attempts (2026-11-08, 2026-11-17,
  2026-11-19) the Gemini route has never sustained the
  several-hundred-call scoring run, and FREE-ONLY bars buying
  quota. The instrument is therefore switched below; the
  Gemini route is not tried again under this registration.

Part 2: the switch, registered before any local run.

- Judge LLM: glm4:9b (GLM-4-9B chat, 9.4B parameters, Q4_0;
  Ollama digest
  5b699761eca535dc55047ad9d2dbf54e3b8697709419ef78a70503ed4bfbcf44),
  served by the resident Ollama on the R610 at
  localhost:11434, reached from this VM through an SSH
  local-forward tunnel. Sampling temperature 0, mirroring
  gemini_client.generate_text's default.
- Embeddings (used by ResponseRelevancy only):
  nomic-embed-text (137M, F16; Ollama digest
  0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f),
  served by the same Ollama, pulled free on 2026-11-19.
- What does not change: the frozen samples
  (results/rag_samples.json), the four Ragas metrics with
  their prompts and rubric, raise_exceptions False with NaN
  counted as a per-question scoring failure and never dropped,
  the H3 and H4 pass lines in PREREGISTRATION.md, and the
  verdict functions in scripts/make_results.py, which compute
  the local verdicts unchanged (imported, not copied).
- What changes: only the judge instrument (LLM and
  embeddings), the scorer entry point
  (src/score_ragas_local.py, a mirror of src/score_ragas.py
  using src/ragas_wrappers_local.py), the results generator
  (scripts/make_results_local.py), and the output paths:
  results/ragas_per_question_local.csv,
  results/ragas_per_question_local.json,
  results/ragas_summary_local.json,
  results/verdicts_local.json, and RESULTS_LOCAL.md. The
  Gemini-instrument files (results/ragas_per_question.*,
  results/verdicts.json, RESULTS.md) are not touched.
- Labelling: every verdict from this run is a LOCAL-JUDGE
  verdict (D9), recorded under that label in the outcome
  entry below, in results/verdicts_local.json, and in
  RESULTS_LOCAL.md. Local-judge verdicts are never presented
  as Gemini results; the Gemini-instrument H3 and H4 state
  remains NOT RUN beside them.

D9 addendum (2026-11-19, registered before the restarted
local run it covers): execution venue and one harness
parameter.

- The first local attempt ran the scorer on this VM against
  the R610's Ollama through an SSH local-forward tunnel. The
  route dropped repeatedly (results/ollama_call_log.jsonl:
  17 attempts, 8 RemoteDisconnected and 5 URLError against 4
  HTTP 200, two of the 200s being the pre-run smoke calls),
  and no job completed. That attempt was stopped at 0 of 160;
  the scorer writes outputs only on completion, so nothing
  was written and nothing is contaminated.
- The execution of record therefore runs on the R610 itself,
  in a detached python:3.12 container with host networking
  and the repository mounted, OLLAMA_BASE
  http://127.0.0.1:11434: no tunnel in the measurement path.
  Same scorer, same wrappers, same models, same frozen
  samples and rubric.
- Measured judge speed on the R610 (Xeon E5540 host, no GPU):
  a 445-token prompt processed in 1.1 s; generation about
  0.9 tokens per second single-stream. The one faster-named
  alternative on the same host, qwen3:4b, was timed on the
  same prompt and did not finish within six minutes (its
  default thinking trace runs long), so it was not
  substituted; glm4:9b stands as registered.
- One harness parameter is set in src/ollama_client.py for
  the local instrument only: num_predict 1024. The rubric's
  judge outputs are short JSON; the cap only bounds a
  runaway generation (the local server otherwise generates
  to the 4,096-token context). Like D4's retry-count change,
  this is a harness parameter, not a protocol term: prompts,
  rubric, samples, and verdict rules are untouched.
