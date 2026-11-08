"""Run the system under test over the frozen question set.

The system is rag-foundation's Build RagPipeline at its
defaults, vendored verbatim in src/ragref/. For each question
in data/questions.json this writes one record to
results/rag_samples.json: the pipeline's extractive answer and
the texts of the reranked top-3 chunks it was built from (the
retrieved contexts Ragas scores against). Fully local and
deterministic; no model is called here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragref.corpus import DOCUMENTS  # noqa: E402
from ragref.pipeline import RagPipeline  # noqa: E402


def main() -> None:
    questions = json.loads((ROOT / "data" / "questions.json").read_text())
    pipeline = RagPipeline(DOCUMENTS)
    chunk_text = {c.chunk_id: c.text for c in pipeline.chunks}
    records = []
    for q in questions:
        result = pipeline.query(q["question"])
        top = result["stages"]["reranked"][:3]
        contexts = [chunk_text[e["chunk_id"]] for e in top]
        records.append({
            "id": q["id"],
            "slice": q["slice"],
            "category": q["category"],
            "question": q["question"],
            "expected_doc": q["expected_doc"],
            "reference": q["reference"],
            "answer": result["answer"],
            "answer_empty": result["answer"] == "",
            "contexts": contexts,
            "context_chunk_ids": [e["chunk_id"] for e in top],
            "top1_doc": top[0]["doc_id"] if top else None,
        })
    out = ROOT / "results" / "rag_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    hits = sum(1 for r in records
               if r["slice"] == "in_corpus" and r["top1_doc"] == r["expected_doc"])
    empty = sum(1 for r in records if r["answer_empty"])
    print(f"wrote {out}: {len(records)} records; in-corpus top-1 doc "
          f"{hits}/28; empty answers {empty}")


if __name__ == "__main__":
    main()
