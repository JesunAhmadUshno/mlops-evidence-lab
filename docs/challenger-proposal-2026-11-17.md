# Challenger proposal: challenger-ensemble (PROPOSAL, not registered, not run)

Date: 2026-11-17. Status: prepared for the Owner's tap. Nothing in
this file is registered, trained, or gated. DECISIONS.md is
unchanged. If the Owner approves, the amendment text in section 4
is appended to DECISIONS.md verbatim as D7 before any run, and the
protocol in section 5 is followed exactly.

## 1. Why H2 is open

H2 is INCONCLUSIVE under the frozen logic in PREREGISTRATION.md:
the promotion gate blocked both candidates it was given, so its
promote direction was never exercised. The tuned challenger
(HistGradientBoosting, max_iter 1000, learning_rate 0.02,
max_leaf_nodes 31, min_samples_leaf 30, l2_regularization 1.0)
scored AUC 0.7756 against the frozen bar of champion + 0.003 =
0.7840, and passed Brier (0.1363 against a cap of 0.1375). The
negative control blocked on both criteria, as designed. The
repository declined to re-tune the challenger after the fact;
exercising the promote direction therefore needs a new candidate
registered by amendment, which is the Owner's call. This file
prepares that amendment.

## 2. The candidate: challenger-ensemble

One candidate, an equal-weight soft-voting ensemble of three
pipelines, each carrying the frozen preprocessing (median
imputation plus scaling for numerics, one-hot for categoricals,
fit on the training split only) and each fit on the same frozen
split (stratified 70/30, random_state 42; 21,000 train, 9,000
test). The ensemble's score is the arithmetic mean of the three
members' predicted probabilities. Weights are 1/3 each, fixed
here, never fitted.

- Member A, the champion configuration:
  HistGradientBoostingClassifier, max_iter 300, learning_rate
  0.06, max_depth 3, random_state 42. Measured on this split:
  AUC 0.7810, KS 0.4196, Brier 0.1355.
- Member B, XGBoost, the tuned configuration of the sibling
  study xgboost-credit-benchmark: n_estimators 200,
  learning_rate 0.04, max_depth 3, min_child_weight 10,
  subsample 0.8, colsample_bytree 0.8, reg_lambda 0.1,
  random_state 42. Measured on this identical split (that
  study's control run reproduced the champion exactly, delta
  0.0000): AUC 0.7822, KS 0.4196, Brier 0.1351.
- Member C, LightGBM, the tuned configuration of the same
  sibling study: n_estimators 200, learning_rate 0.04,
  num_leaves 15, min_child_samples 40, subsample 1.0,
  subsample_freq 1, colsample_bytree 0.8, reg_lambda 1.0,
  random_state 42. Measured on this identical split: AUC
  0.7816, KS 0.4239, Brier 0.1348.

## 3. Why this candidate, on the evidence

The registered challenger failed in a specific way: it was a
single model from the champion's own family, pushed toward
regularization. It bought the field's second-best KS (0.4235)
and paid for it in AUC (0.7756, below the champion it had to
beat by 0.003). The negative control was designed to fail. So
the promote direction has never been tried by a candidate built
to clear the AUC bar.

A second single model is the weak option, and the record says
so. The sibling benchmark ran the tuned singles on this exact
split: XGBoost 0.7822 (+0.0012 over the champion), LightGBM
0.7816 (+0.0006), a tuned sklearn HistGB variant 0.7812
(+0.0002). Every one falls short of the +0.0030 the gate
requires, and the sibling's paired bootstrap put the XGBoost
delta at +0.0011 with a 95 percent interval of [-0.0006,
+0.0029], which does not reach the bar either. Picking any one
of them as the new challenger would repeat a measured failure
mode with a new name.

The untried axis on this record is diversity. The three members
are three different implementations with different inductive
biases: depth-limited histogram boosting, XGBoost with row and
column subsampling, and LightGBM's leaf-wise growth. Their
errors are therefore only partly shared, and equal-weight
probability averaging of near-equal rankers is the standard
variance-reduction step: it can only be taken before the run,
with the members and weights frozen, which is what section 4
does. The Brier criterion is the low-risk side: all three
members already pass it individually (0.1348 to 0.1355 against
the 0.1375 cap), so the ensemble starts from three passing
Brier positions and three AUCs within 0.0022 of the bar.

Stated plainly, the uncertainty: the ensemble's AUC gain is
plausible, not established. No record in this programme has
measured an ensemble on this split. A gain of +0.0030 over the
champion is inside the range probability averaging produces on
tabular credit data, and it is also possible the ensemble lands
at 0.782 to 0.783 and blocks. If it blocks, H2 stays
INCONCLUSIVE, the result is published either way, and this line
of attack is spent: the amendment authorizes one candidate and
one run, with no re-weighting, member swap, or second ensemble
behind it.

## 4. The exact amendment text (to be appended to DECISIONS.md as D7 on the Owner's tap)

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

## 5. The exact gate protocol the promotion test would follow

The frozen gate, unchanged, in the frozen order of operations:

1. Train challenger-ensemble once (section 4), producing its
   single MLflow run with AUC, KS, and Brier on the frozen
   test split.
2. Run the existing gate once against it:
   .venv/bin/python src/gate.py challenger-ensemble
   src/gate.py reads the candidate's metrics from MLflow and
   the champion's from the champion-rerun run (AUC 0.7810,
   Brier 0.1355). It promotes if and only if candidate AUC is
   at least champion AUC + 0.003 (0.7840) and candidate Brier
   is at most champion Brier + 0.002 (0.1375). No other input,
   no hand-entered numbers.
3. The gate appends its decision, with the metric table and
   per-criterion reasons, to results/gate_decisions.json. On a
   promote, the gate registers the candidate's model as a new
   version of credit_pd_champion and moves the champion alias
   to it, exactly as the frozen protocol specifies.
4. Regenerate the record: scripts/make_results.py, and the
   README headline if the verdict changes.

Verdict mapping, fixed before the run: if the gate promotes
challenger-ensemble, the promote direction stands exercised
with a qualifying candidate, the negative control stands
blocked, and H2 resolves CONFIRMED. If the gate blocks
challenger-ensemble, H2 remains INCONCLUSIVE and the blocked
metrics are published in RESULTS.md like every other number in
this repository. No further candidate follows from this
amendment in either case.
