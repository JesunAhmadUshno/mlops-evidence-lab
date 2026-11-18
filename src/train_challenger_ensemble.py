"""Half 1 amendment run: challenger-ensemble, registered in
DECISIONS.md D7 (Owner's tap, 2026-11-17) before this script
existed. One candidate, one run.

The candidate is an equal-weight (1/3 each, fixed, never
fitted) soft-voting average of three pipelines, each carrying
the frozen preprocessing (make_pipe) and each fit on the
frozen split (load_split):

- A: the champion configuration, HistGradientBoosting
  (max_iter 300, learning_rate 0.06, max_depth 3, seed 42).
- B: XGBoost at the sibling xgboost-credit-benchmark tuned
  configuration (its pins: xgboost==3.4.1).
- C: LightGBM at the sibling tuned configuration
  (lightgbm==4.7.0).

sklearn's VotingClassifier with voting="soft" and equal
weights computes exactly the arithmetic mean of the members'
predicted probabilities (verified against a manual mean in
pre-run validation: max difference 0.0). Metrics come from
the existing metrics_for on the frozen test split and are
logged with the full parameter set, as in
src/train_candidates.py. The frozen four candidates are not
re-run; the gate asserts exactly one run per name.

The model artifact is logged with cloudpickle serialization:
skops, used for the pure-sklearn candidates, refuses the
XGBoost and LightGBM boosters even with trusted types named
(pre-run validation, DECISIONS.md D7 result entry). The
serialization format is a logging detail; the registered
metrics, parameters, and gate are unaffected.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier, VotingClassifier,
)
from xgboost import XGBClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from credit_common import (  # noqa: E402
    DATA, dump_json, load_split, make_pipe, metrics_for,
)

EXPERIMENT = "credit-risk-lab-model-line"
NAME = "challenger-ensemble"


def member_models() -> dict:
    return {
        "a": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=3, random_state=42),
        "b": XGBClassifier(
            n_estimators=200, learning_rate=0.04, max_depth=3,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=0.1, random_state=42),
        "c": LGBMClassifier(
            n_estimators=200, learning_rate=0.04, num_leaves=15,
            min_child_samples=40, subsample=1.0, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42),
    }


def params_of(model) -> dict:
    return {k: (str(v) if not isinstance(v, (int, float, bool)) or v is None else v)
            for k, v in model.get_params().items()}


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    Xtr, Xte, ytr, yte = load_split()
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    models = member_models()
    voting = VotingClassifier(
        estimators=[(k, make_pipe(m)) for k, m in models.items()],
        voting="soft", weights=[1, 1, 1])
    started = time.time()
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.set_tag("candidate", NAME)
        mlflow.set_tag("data_sha256", data_sha)
        params = {
            "model_class": ("VotingClassifier(soft, weights 1/3 each) of "
                            "HistGradientBoostingClassifier + XGBClassifier "
                            "+ LGBMClassifier, each in the frozen make_pipe "
                            "preprocessing"),
            "voting": "soft",
            "weights": "1/3, 1/3, 1/3 (fixed, never fitted)",
            "split": "stratified 70/30, random_state 42",
            "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
            "labels": "as published",
        }
        for key, model in models.items():
            params[f"member_{key}_model_class"] = type(model).__name__
            for p, v in params_of(model).items():
                params[f"member_{key}_{p}"] = v
        mlflow.log_params(params)
        pipe = voting.fit(Xtr, ytr)
        p_test = pipe.predict_proba(Xte)[:, 1]
        p_train = pipe.predict_proba(Xtr)[:, 1]
        m = metrics_for(np.asarray(yte), p_test, p_train)
        for key in ("auc", "ks", "gini", "brier", "score_psi"):
            mlflow.log_metric(key, m[key])
        calib_path = ROOT / "results" / f"calibration_{NAME}.json"
        dump_json(m["calibration"], calib_path)
        mlflow.log_artifact(str(calib_path))
        mlflow.sklearn.log_model(pipe, name="model",
                                 serialization_format="cloudpickle")
        run_id = run.info.run_id
    print(f"{NAME}: AUC {m['auc']} KS {m['ks']} Brier {m['brier']} "
          f"(run {run_id}, {round(time.time() - started, 2)}s)")


if __name__ == "__main__":
    main()
