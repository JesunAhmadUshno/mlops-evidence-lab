"""Score the frozen samples with Ragas over the Gemini route.

Reads results/rag_samples.json, builds one SingleTurnSample per
question, and runs ragas.evaluate with faithfulness, answer
relevancy, context precision (with reference), and context
recall, using the wrappers in src/ragas_wrappers.py. Ragas runs
with raise_exceptions False: a NaN is a scoring failure for that
question and metric, counted and published, never dropped.

Outputs: results/ragas_per_question.csv (the full table) and
results/ragas_summary.json (means overall and per slice, plus
per-metric scored and failed counts).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ragas_compat  # noqa: E402

ragas_compat.apply()

import pandas as pd  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

from ragas_wrappers import GeminiEmbeddings, GeminiLLM  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def main() -> None:
    records = json.loads((ROOT / "results" / "rag_samples.json").read_text())
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["reference"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)
    llm = GeminiLLM()
    embeddings = GeminiEmbeddings()
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=900, max_retries=2, max_wait=120,
                             max_workers=4),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    rows = []
    for record, (_, scored) in zip(records, frame.iterrows()):
        row = {k: record[k] for k in
               ("id", "slice", "category", "question", "expected_doc",
                "reference", "answer", "answer_empty", "top1_doc")}
        row["contexts"] = record["contexts"]
        for metric in METRICS:
            value = scored.get(metric)
            row[metric] = None if is_nan(value) else float(value)
        rows.append(row)
    out_csv = ROOT / "results" / "ragas_per_question.csv"
    flat = [{k: v for k, v in r.items() if k != "contexts"} for r in rows]
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    (ROOT / "results" / "ragas_per_question.json").write_text(
        json.dumps(rows, indent=1))

    def mean(rows_, metric):
        vals = [r[metric] for r in rows_ if r[metric] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {"n": len(rows), "metrics": {}}
    for metric in METRICS:
        scored_n = sum(1 for r in rows if r[metric] is not None)
        summary["metrics"][metric] = {
            "overall_mean": mean(rows, metric),
            "in_corpus_mean": mean(
                [r for r in rows if r["slice"] == "in_corpus"], metric),
            "adversarial_mean": mean(
                [r for r in rows if r["slice"] == "adversarial"], metric),
            "scored": scored_n,
            "failed": len(rows) - scored_n,
        }
    summary["empty_answers"] = {
        "in_corpus": sum(1 for r in rows
                          if r["slice"] == "in_corpus" and r["answer_empty"]),
        "adversarial": sum(1 for r in rows
                            if r["slice"] == "adversarial" and r["answer_empty"]),
    }
    (ROOT / "results" / "ragas_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
