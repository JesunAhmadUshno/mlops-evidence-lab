# Pre-registration: tracked model lines and published RAG evaluation failures on a local stack

Locked before any measured run exists. The commit that adds this
file precedes every result in this repository; git history proves
the order. Any change to this file after locking is an amendment
and must be recorded in DECISIONS.md with its reason, before the
runs it affects.

## Research question

Can the two pieces of MLOps evidence a hiring panel actually
asks for be produced honestly on a local stack: (1) a tracked,
registered, reproducible model line whose published metrics
survive a from-scratch rerun under experiment tracking, and
(2) a standardized RAG evaluation whose failure cases are
published, not hidden?

## Machine of record

All local runs happen on one named machine: the Muse VM,
hostname htch-runtime (Linux 7.0.0-43-generic, AMD EPYC 9D25,
4 vCPUs, 7 GiB RAM). Exact library versions are recorded in the
run manifests. No number from any other machine is reported in
this study. The Ragas judge and embedding models run remotely
by definition; their calls are made from this machine through
the route frozen below.

## Half 1: the credit-risk model line under MLflow

### Subject and data (frozen)

- Subject repository: JesunAhmadUshno/credit-risk-lab, pinned
  at commit fee580d0beee43c76dc57c6c366c211533054956 (the commit
  its own results/run_manifest.json records). The subject repo
  is cloned read-only by scripts/fetch_credit_data.sh and is
  never modified.
- Data file: data/default of credit card clients.xls from that
  commit (UCI default of credit card clients, 30,000 accounts),
  SHA-256
  30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933,
  verified by the fetch script against the subject repo's own
  published manifest value before any run uses it.
- Published champion figures under test (from the subject
  repo's results/results.json at the pinned commit): gradient
  boosting AUC 0.7810, KS 0.4196, Brier 0.1355.

### The rerun protocol (frozen)

The rerun re-implements the subject repo's champion training
exactly as its src/credit_risk_lab.py specifies, changing only
the tracking layer:

- Load with pandas read_excel, header=1; rename the target to
  default; EDUCATION values 0, 5, 6 mapped to 4; MARRIAGE 0
  mapped to 3.
- Features, exactly as the subject: utilization, pay_ratio
  (filled 0, clipped to [-1, 5]), delinq_mean, delinq_max,
  bill_trend, dpd_months added to the raw fields; model inputs
  are LIMIT_BAL, AGE, BILL_AMT1, PAY_AMT1, utilization,
  pay_ratio, delinq_mean, delinq_max, bill_trend, PAY_0, PAY_2,
  PAY_3, PAY_4, PAY_5, PAY_6 (numeric) and SEX, EDUCATION,
  MARRIAGE (categorical).
- Preprocessing inside one sklearn Pipeline: median imputation
  plus standard scaling for numerics, one-hot encoding
  (handle_unknown=ignore) for categoricals. The pipeline is fit
  on the training split only. That is the subject's
  leakage-safe protocol: raw fields are the account's own
  history up to the observation point, the target is default
  in the following month, and no statistic is fit on test rows.
- Split: train_test_split, test_size 0.30, stratify on the
  target, random_state 42. Training rows 21,000, test rows
  9,000.
- Models, exactly as the subject: logistic regression
  (max_iter 1000, C 0.1) and gradient boosting
  (HistGradientBoostingClassifier, max_iter 300,
  learning_rate 0.06, max_depth 3, random_state 42). The
  gradient boosting run is the champion rerun.
- Metrics per run, computed on the frozen test split: AUC, KS
  (the subject's ks_stat), Gini (2 x AUC - 1), Brier score,
  score PSI (train vs test, 10 bins), and the 10-bin
  calibration table. Rounding follows the subject: 4 decimals.

### Tracking and registry (frozen)

- MLflow with a local sqlite backend store and a local artifact
  directory, both gitignored. No server, no remote tracking.
- Experiment name: credit-risk-lab-model-line. One MLflow run
  per candidate below; each run logs the full parameter set,
  the metrics above, the calibration table as a JSON artifact,
  and the fitted pipeline as an sklearn model artifact.
- Candidates:
  - champion-rerun: the gradient boosting configuration above.
    (The logistic regression run is also logged, as in the
    subject, but is not a registry candidate.)
  - challenger-tuned: same features and preprocessing,
    HistGradientBoostingClassifier with max_iter 1000,
    learning_rate 0.02, max_leaf_nodes 31, min_samples_leaf 30,
    l2_regularization 1.0, early_stopping False,
    random_state 42. The design input is the subject repo's own
    published evidence (its 5-seed champion mean AUC 0.7830 in
    results/rigor.json); no run of this lab precedes this
    registration.
  - negative-control: the champion configuration trained on
    training labels permuted with numpy RandomState(123)
    (test labels untouched). A deliberately degraded model:
    its test AUC is expected near 0.5.
- Registry: model name credit_pd_champion. The champion rerun
  is registered as version 1 and given the alias champion
  before any gate evaluation.

### The promotion gate (frozen)

The gate is a script (src/gate.py) that reads candidate metrics
from MLflow and applies this rule; it takes no other input.

A candidate is promoted if and only if, on the frozen test
split:
- candidate AUC is at least champion AUC + 0.003, and
- candidate Brier is at most champion Brier + 0.002,
where champion means the champion-rerun run's metrics.

Promotion means: the candidate run's model is registered as a
new version of credit_pd_champion and the alias champion moves
to it. Every evaluation, pass or fail, is written to
results/gate_decisions.json with the metric table and the
reason. The gate is run once on challenger-tuned and once on
negative-control, in that order.

### H1 and H2 (frozen, with pass lines)

Verdict vocabulary: CONFIRMED, REFUTED, INCONCLUSIVE.

H1 (reproduction): CONFIRMED if the champion rerun's AUC
differs from the published 0.7810 by at most 0.005 in absolute
value. REFUTED if it differs by more. INCONCLUSIVE only if the
rerun cannot be executed at all. KS and Brier deltas are
reported beside the verdict but do not decide it.

H2 (the gate works in both directions): let C be the
challenger-tuned outcome and N the negative-control outcome.
- CONFIRMED if the gate promotes challenger-tuned (which
  requires its metrics to clear the frozen bar) and blocks
  negative-control.
- REFUTED if the gate promotes negative-control, or if
  challenger-tuned clears the frozen bar on metrics but the
  gate does not promote it (the mechanism, not the model,
  failed).
- INCONCLUSIVE if challenger-tuned does not clear the frozen
  bar and the gate blocks it: the gate then behaved correctly
  in the only direction exercised, but the promote direction
  was never tested with a qualifying candidate. The metrics
  are published either way.

## Half 2: Ragas evaluation of the rag-foundation pipeline

### Subject and the system under test (frozen)

- Subject repository: JesunAhmadUshno/rag-foundation, pinned at
  commit 007e33238e6ce49100650b30f6820149410f022b. Its
  level-2-build corpus.py and pipeline.py are vendored verbatim
  into src/ragref/ of this repo with a provenance note (the
  subject is Apache-2.0). The subject repo is never modified.
- The system under test is the Build pipeline's public API:
  RagPipeline over the 20-document corpus with the defaults
  chunk_size 40, overlap 10, answering with query(question)
  at its defaults (top_k 3, max_sentences 2). The answer is the
  pipeline's extractive answer text. The retrieved contexts
  for scoring are the texts of the reranked top-3 chunks the
  answer was built from (the reranked stage's first 3 entries,
  in order). The system is deterministic and local: it makes
  no model calls. Every model call in Half 2 belongs to the
  judge or the embeddings.

### Question set (frozen)

40 questions, committed as data/questions.json by
scripts/build_questions.py from committed source files before
any scoring run:

- 28 in-corpus questions: the 24 regression questions and the
  4 paraphrase questions p01-condensation, p02-molten,
  p05-storytelling, p06-byproduct from the subject's
  level-3-deep/questions_deep.json at the pinned commit, ids
  and expected documents unchanged. Reference answers are in
  data/reference_answers.json, written from the corpus text
  before the run.
- 12 adversarial questions (30 percent, the named share),
  listed verbatim in data/adversarial.json with a type each:
  4 absent_detail (facts the corpus never states), 4
  false_premise (questions whose premise the corpus
  contradicts or does not support), 4 out_of_corpus (topics
  absent from the corpus). They are designed to fail: the
  reference answer for every adversarial question is the
  frozen string "Not answerable from the corpus."
- No question is dropped after the run for any reason. A
  question the system answers with the empty string is
  recorded as an empty answer, tallied separately, and stays
  in every table.

### Scoring (frozen)

- Library: Ragas, version pinned in requirements.txt. Metrics:
  faithfulness, answer relevancy (Ragas response relevancy),
  context precision (LLM context precision with reference),
  context recall (LLM context recall). Each question becomes
  one SingleTurnSample with user_input, retrieved_contexts,
  response, and reference as defined above.
- Judge LLM and embeddings reach Gemini only through the
  existing connected Gemini connector (custom.legacy-gemini):
  requests go to generativelanguage.googleapis.com with the
  credential attached by the connector's dynamic-credentials
  helper under the stored name custom.gemini, exactly as the
  connector documents. No other host, no other provider, no
  new key or account.
  - Judge model: gemini-3-flash-preview via generateContent,
    temperature 0.
  - Embedding model: gemini-embedding-001 via embedContent
    and batchEmbedContents.
- Pacing and retries (frozen, part of the protocol): a single
  client-side limiter keeps at least 4.0 seconds between
  successive requests to the route. On HTTP 429 or any 5xx,
  a request is retried up to 4 times with backoff of 15, 30,
  60, 120 seconds. Every request, retry, and response status
  is logged to results/gemini_call_log.jsonl.
- Failures: Ragas runs with raise_exceptions False. A metric
  that returns NaN for a question is a scoring failure for
  that question and metric: it is counted, listed in the
  failure table, and excluded from that metric's means. A
  question is never silently dropped. The full per-question
  table, scores and failures alike, is a deliverable and is
  published in results/ragas_per_question.csv and in
  RESULTS.md.

### H3 and H4 (frozen, with pass lines)

H3 (the required hypothesis: faithfulness on the adversarial
slice): let F_adv be the mean faithfulness over adversarial
questions that received a faithfulness score. An adversarial
question whose system answer is empty receives no faithfulness
score from Ragas; it is tallied as correctly silent and is not
in F_adv.
- If fewer than 9 of the 12 adversarial questions receive a
  faithfulness score, H3 is INCONCLUSIVE regardless of values.
- CONFIRMED if F_adv is below 0.50.
- REFUTED if F_adv is at least 0.50.

H4 (where the failure shows instead): let R_in and R_adv be
the mean answer relevancy over scored in-corpus and
adversarial questions. If either slice has fewer than 75
percent of its questions scored for relevancy, H4 is
INCONCLUSIVE.
- CONFIRMED if R_in - R_adv is at least 0.30.
- REFUTED if R_in - R_adv is at most 0.05.
- INCONCLUSIVE between.

Overall means for all four metrics, and per-slice means, are
reported in RESULTS.md beside the verdicts.

## Results generation (frozen)

RESULTS.md is generated by scripts/make_results.py from the
recorded run artifacts (results/half1_metrics.json,
results/gate_decisions.json, results/ragas_per_question.csv);
verdicts are computed by that script from the frozen pass
lines above, never typed by hand. With no runs, the generator
produces an awaiting-runs RESULTS.md that cannot be mistaken
for results. README headline numbers are copied from the
generated RESULTS.md, and AUDIT.md records the Jev claim_check
of the headline verdicts with model, noul, and route.
