"""The registry promotion gate, exactly as frozen.

Reads candidate metrics from MLflow (never from hand-entered
numbers) and applies the registered rule: promote if and only
if candidate AUC is at least champion AUC + 0.003 and candidate
Brier is at most champion Brier + 0.002, champion being the
champion-rerun run. Promotion registers the candidate's model
as a new version of credit_pd_champion and moves the alias
champion to it. Every decision lands in
results/gate_decisions.json.

Usage: python src/gate.py [candidate ...]
Default order (the registered order): challenger-tuned, then
negative-control.
"""
import json
import sys
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_NAME = "credit_pd_champion"
AUC_MARGIN = 0.003
BRIER_SLACK = 0.002
DEFAULT_ORDER = ["challenger-tuned", "negative-control"]


def run_metrics(client, experiment_id, run_name) -> dict:
    runs = client.search_runs([experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
    assert len(runs) == 1, f"expected one run named {run_name}, found {len(runs)}"
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "auc": run.data.metrics["auc"],
        "ks": run.data.metrics["ks"],
        "brier": run.data.metrics["brier"],
    }


def decide(champion: dict, candidate: dict) -> tuple[bool, list[str]]:
    reasons = []
    auc_ok = candidate["auc"] >= champion["auc"] + AUC_MARGIN
    reasons.append(
        f"AUC {candidate['auc']:.4f} vs required >= "
        f"{champion['auc'] + AUC_MARGIN:.4f} (champion {champion['auc']:.4f} + {AUC_MARGIN}): "
        f"{'pass' if auc_ok else 'fail'}")
    brier_ok = candidate["brier"] <= champion["brier"] + BRIER_SLACK
    reasons.append(
        f"Brier {candidate['brier']:.4f} vs required <= "
        f"{champion['brier'] + BRIER_SLACK:.4f} (champion {champion['brier']:.4f} + {BRIER_SLACK}): "
        f"{'pass' if brier_ok else 'fail'}")
    return auc_ok and brier_ok, reasons


def main(candidates) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("credit-risk-lab-model-line")
    assert experiment is not None, "experiment not found; run train_candidates.py first"
    champion = run_metrics(client, experiment.experiment_id, "champion-rerun")
    decisions_path = ROOT / "results" / "gate_decisions.json"
    decisions = []
    if decisions_path.exists():
        decisions = json.loads(decisions_path.read_text())
    for name in candidates:
        cand = run_metrics(client, experiment.experiment_id, name)
        before = client.get_model_version_by_alias(REGISTRY_NAME, "champion")
        promote, reasons = decide(champion, cand)
        entry = {
            "candidate": name,
            "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "champion_metrics": {k: champion[k] for k in ("auc", "ks", "brier")},
            "candidate_metrics": {k: cand[k] for k in ("auc", "ks", "brier")},
            "candidate_run_id": cand["run_id"],
            "bar": {"auc_margin": AUC_MARGIN, "brier_slack": BRIER_SLACK},
            "decision": "promote" if promote else "block",
            "reasons": reasons,
            "alias_champion_before": str(before.version),
            "alias_champion_after": str(before.version),
            "registered_version": None,
        }
        if promote:
            version = mlflow.register_model(f"runs:/{cand['run_id']}/model", REGISTRY_NAME)
            client.set_registered_model_alias(REGISTRY_NAME, "champion", version.version)
            entry["registered_version"] = str(version.version)
            entry["alias_champion_after"] = str(version.version)
        decisions.append(entry)
        print(f"{name}: {entry['decision']} ({'; '.join(reasons)})")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(json.dumps(decisions, indent=1))
    print(f"wrote {decisions_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_ORDER)
