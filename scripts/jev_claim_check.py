"""Run the Jev claim_check gate on the headline verdicts.

Builds the claim from results/verdicts.json and the generated
RESULTS.md numbers (never hand-typed), writes the spec to
results/jev-claim-spec.json, calls the TypeSafe connector
helper (the vault flow; the key never appears here), and saves
the raw response to results/jev-claim-check.json. The route
bands are the house claim_check bands: clear at noul 0.85 or
above, block at 0.10 or below, review between.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path.home() / "workspace/custom-connectors/custom.typesafe/bin/typesafe_jev.py"


def main() -> None:
    verdicts = json.loads((ROOT / "results" / "verdicts.json").read_text())
    half1 = json.loads((ROOT / "results" / "half1_metrics.json").read_text())
    summary = json.loads((ROOT / "results" / "ragas_summary.json").read_text())
    champ = half1["champion-rerun"]
    claim = (
        f"Under one pre-registered protocol (PREREGISTRATION.md is the first "
        f"commit of the repository; verdicts computed by "
        f"scripts/make_results.py from the recorded artifacts): "
        f"H1 is {verdicts['H1']}: the MLflow-tracked rerun of "
        f"credit-risk-lab's champion reproduces its published test AUC of "
        f"0.7810 with {champ['auc']:.4f} (KS {champ['ks']:.4f} and Brier "
        f"{champ['brier']:.4f} also match the published figures exactly). "
        f"H2 is {verdicts['H2']}: the registry promotion gate blocked the "
        f"tuned challenger (AUC {half1['challenger-tuned']['auc']:.4f}, "
        f"below the frozen bar of champion + 0.003) and blocked the "
        f"label-permuted negative control (AUC "
        f"{half1['negative-control']['auc']:.4f}); no candidate qualified "
        f"for promotion, so the promote direction of the gate is untested. "
        f"H3 is {verdicts['H3']} and H4 is {verdicts['H4']} for the Ragas "
        f"evaluation of rag-foundation's extractive pipeline on 40 frozen "
        f"questions (28 in-corpus, 12 adversarial): adversarial mean "
        f"faithfulness {summary['metrics']['faithfulness']['adversarial_mean']}, "
        f"answer relevancy {summary['metrics']['answer_relevancy']['in_corpus_mean']} "
        f"in-corpus against {summary['metrics']['answer_relevancy']['adversarial_mean']} "
        f"adversarial, with every per-question score and failure published."
    )
    evidence = (ROOT / "RESULTS.md").read_text()
    spec = {
        "model": "jev-1.13.0",
        "state": {"claim": claim, "evidence": evidence},
        "questions": {
            "supported": {
                "type": "noul",
                "instructions": "Does the `evidence` in state fully support the `claim` in state, with no part of the claim going beyond the evidence?",
                "criteria": {
                    "true": "Every part of the claim is directly backed by the evidence",
                    "false": "Any part of the claim is missing from, contradicted by, or stronger than the evidence",
                },
            }
        },
    }
    spec_path = ROOT / "results" / "jev-claim-spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    proc = subprocess.run(
        [sys.executable, str(HELPER), "evaluate", str(spec_path)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("jev helper failed")
    body = json.loads(proc.stdout)
    (ROOT / "results" / "jev-claim-check.json").write_text(
        json.dumps(body, indent=1))
    answers = body.get("answers", {})
    noul = answers.get("supported", {}).get("noul")
    route = ("clear" if noul is not None and noul >= 0.85 else
             "block" if noul is not None and noul <= 0.10 else "review")
    print(json.dumps({
        "model": body.get("model"), "noul": noul, "route": route,
        "usage": body.get("usage"),
    }, indent=1))


if __name__ == "__main__":
    main()
