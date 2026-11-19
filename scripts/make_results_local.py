"""Generate the LOCAL-JUDGE results (DECISIONS.md D9).

Reads results/ragas_per_question_local.json and
results/ragas_summary_local.json, computes the H3 and H4
verdicts with the frozen verdict functions imported unchanged
from scripts/make_results.py, and writes
results/verdicts_local.json and RESULTS_LOCAL.md. Every output
is labelled as a local-judge (D9) result: judge glm4:9b,
embeddings nomic-embed-text, both served by Ollama on the
R610. These are never Gemini results; the Gemini-instrument
RESULTS.md and results/verdicts.json are not touched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_results import fmt, verdict_h3, verdict_h4  # noqa: E402

JUDGE = ("local judge (DECISIONS.md D9): glm4:9b via Ollama on "
         "the R610; embeddings nomic-embed-text via the same "
         "Ollama")


def main() -> None:
    rows = json.loads(
        (ROOT / "results" / "ragas_per_question_local.json").read_text())
    summary = json.loads(
        (ROOT / "results" / "ragas_summary_local.json").read_text())
    h3, f_adv, h3_scored, h3_silent = verdict_h3(rows)
    h4, r_in, r_adv, h4_gap = verdict_h4(rows)
    verdicts = {
        "instrument": JUDGE,
        "H3": {"verdict": h3, "f_adv": f_adv,
               "adversarial_scored": h3_scored,
               "adversarial_silent": h3_silent},
        "H4": {"verdict": h4, "r_in": r_in, "r_adv": r_adv,
               "gap": h4_gap},
    }
    (ROOT / "results" / "verdicts_local.json").write_text(
        json.dumps(verdicts, indent=1))
    lines = []
    lines.append("# Results (LOCAL JUDGE, DECISIONS.md D9)")
    lines.append("")
    lines.append("These verdicts were produced by a LOCAL judge, "
                 "not by the registered Gemini instrument: " + JUDGE + ".")
    lines.append("The frozen samples, the four Ragas metrics and "
                 "their rubric, the H3 and H4 pass lines in "
                 "PREREGISTRATION.md, and the verdict functions in "
                 "scripts/make_results.py are unchanged; only the "
                 "judge instrument changed, under D9. The "
                 "Gemini-instrument H3 and H4 state remains NOT RUN "
                 "(see RESULTS.md).")
    lines.append("")
    lines.append("## Half 2 (local judge): verdicts")
    lines.append("")
    lines.append(f"- H3 {h3}: mean faithfulness on the adversarial "
                 f"slice F_adv = {fmt(f_adv)} over {h3_scored} scored "
                 f"questions ({h3_silent} correctly silent); frozen "
                 f"line: CONFIRMED below 0.50 with at least 9 scored.")
    lines.append(f"- H4 {h4}: mean answer relevancy in-corpus "
                 f"R_in = {fmt(r_in)}, adversarial R_adv = {fmt(r_adv)}, "
                 f"gap = {fmt(h4_gap)}; frozen lines: CONFIRMED at a "
                 f"gap of at least 0.30, REFUTED at most 0.05, "
                 f"INCONCLUSIVE between, with at least 75 percent of "
                 f"each slice scored.")
    lines.append("")
    lines.append("## Half 2 (local judge): metric means")
    lines.append("")
    lines.append("| Metric | Overall | In-corpus | Adversarial | Scored | Failed |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for metric, m in summary["metrics"].items():
        lines.append(f"| {metric} | {fmt(m['overall_mean'])} | "
                     f"{fmt(m['in_corpus_mean'])} | "
                     f"{fmt(m['adversarial_mean'])} | {m['scored']} | "
                     f"{m['failed']} |")
    lines.append("")
    lines.append("The full per-question table, scores and failures "
                 "alike, is in results/ragas_per_question_local.csv.")
    lines.append("")
    (ROOT / "RESULTS_LOCAL.md").write_text("\n".join(lines))
    print(json.dumps(verdicts, indent=1))
    print("wrote results/verdicts_local.json and RESULTS_LOCAL.md")


if __name__ == "__main__":
    main()
