"""Half 1 training: run every candidate under MLflow tracking.

Candidates, as frozen in PREREGISTRATION.md:
- logistic-regression: the subject's baseline (logged, not a
  registry candidate).
- champion-rerun: the subject's gradient boosting configuration,
  exactly. After its run, the model is registered as
  credit_pd_champion version 1 and given the alias champion.
- challenger-tuned: a stronger HistGradientBoosting configuration
  on the same features, split, and preprocessing.
- negative-control: the champion configuration trained on
  training labels permuted with RandomState(123).

Outputs: results/half1_metrics.json (metrics and MLflow run ids
per candidate) and results/half1_manifest.json (versions, data
hash, timestamps). The MLflow store itself is local and
gitignored; these JSON files are the committed record.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from credit_common import (  # noqa: E402
    DATA, dump_json, load_split, make_pipe, metrics_for,
)

EXPERIMENT = "credit-risk-lab-model-line"
REGISTRY_NAME = "credit_pd_champion"
SHUFFLE_SEED = 123


def model_configs():
    return {
        "logistic-regression": LogisticRegression(max_iter=1000, C=0.1),
        "champion-rerun": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=3, random_state=42),
        "challenger-tuned": HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.02, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False, random_state=42),
        "negative-control": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=3, random_state=42),
    }


def params_of(model) -> dict:
    return {k: (str(v) if not isinstance(v, (int, float, bool)) or v is None else v)
            for k, v in model.get_params().items()}


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    Xtr, Xte, ytr, yte = load_split()
    ytr_control = np.random.RandomState(SHUFFLE_SEED).permutation(
        np.asarray(ytr))
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    results = {}
    client = MlflowClient()
    for name, model in model_configs().items():
        y_fit = ytr_control if name == "negative-control" else ytr
        started = time.time()
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("candidate", name)
            mlflow.set_tag("data_sha256", data_sha)
            mlflow.log_params({
                "model_class": type(model).__name__,
                **params_of(model),
                "split": "stratified 70/30, random_state 42",
                "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
                "labels": ("permuted, RandomState(123)"
                           if name == "negative-control" else "as published"),
            })
            pipe = make_pipe(model).fit(Xtr, y_fit)
            p_test = pipe.predict_proba(Xte)[:, 1]
            p_train = pipe.predict_proba(Xtr)[:, 1]
            m = metrics_for(np.asarray(yte), p_test, p_train)
            for key in ("auc", "ks", "gini", "brier", "score_psi"):
                mlflow.log_metric(key, m[key])
            calib_path = ROOT / "results" / f"calibration_{name}.json"
            dump_json(m["calibration"], calib_path)
            mlflow.log_artifact(str(calib_path))
            mlflow.sklearn.log_model(
                pipe, name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                ])
            run_id = run.info.run_id
        results[name] = {**m, "run_id": run_id,
                         "seconds": round(time.time() - started, 2)}
        print(f"{name}: AUC {m['auc']} KS {m['ks']} Brier {m['brier']}")
    champ_run = results["champion-rerun"]["run_id"]
    version = mlflow.register_model(f"runs:/{champ_run}/model", REGISTRY_NAME)
    client.set_registered_model_alias(REGISTRY_NAME, "champion", version.version)
    results["registry"] = {
        "name": REGISTRY_NAME,
        "champion_version": str(version.version),
        "champion_run_id": champ_run,
        "alias": "champion",
    }
    dump_json(results, ROOT / "results" / "half1_metrics.json")
    import pandas as pd
    import sklearn
    manifest = {
        "experiment": EXPERIMENT,
        "tracking_uri": "sqlite:///mlflow.db (local, gitignored)",
        "data_file": "data/credit_card.xls",
        "data_sha256": data_sha,
        "python": platform.python_version(),
        "mlflow": mlflow.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True).stdout.strip(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dump_json(manifest, ROOT / "results" / "half1_manifest.json")
    print("registered", REGISTRY_NAME, "version", version.version,
          "with alias champion")


if __name__ == "__main__":
    main()
