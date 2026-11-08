"""Build data/questions.json, the frozen 40-question set of Half 2.

Sources (all committed or pinned before any scoring run):
- the 24 regression and 4 named paraphrase questions from
  rag-foundation level-3-deep/questions_deep.json at the pinned
  commit (clone produced by scripts/fetch_rag_subject.sh),
- data/reference_answers.json (reference answers written from
  the corpus text),
- data/adversarial.json (the 12 adversarial questions).

Output rows carry: id, slice (in_corpus or adversarial),
category (the subject's category, or the adversarial type),
question, expected_doc (in-corpus only), reference.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "data" / "ref" / "rag-foundation" / "level-3-deep" / "questions_deep.json"
PARAPHRASE_KEEP = {"p01-condensation", "p02-molten", "p05-storytelling", "p06-byproduct"}
ADVERSARIAL_REFERENCE = "Not answerable from the corpus."


def main() -> None:
    deep = json.loads(DEEP.read_text())
    refs = json.loads((ROOT / "data" / "reference_answers.json").read_text())
    adv = json.loads((ROOT / "data" / "adversarial.json").read_text())
    rows = []
    for item in deep:
        keep = item["category"] == "regression" or (
            item["category"] == "paraphrase" and item["id"] in PARAPHRASE_KEEP
        )
        if not keep:
            continue
        rows.append({
            "id": item["id"],
            "slice": "in_corpus",
            "category": item["category"],
            "question": item["question"],
            "expected_doc": item["expected_doc"],
            "reference": refs[item["id"]],
        })
    for item in adv:
        rows.append({
            "id": item["id"],
            "slice": "adversarial",
            "category": item["type"],
            "question": item["question"],
            "expected_doc": None,
            "reference": ADVERSARIAL_REFERENCE,
        })
    assert len(rows) == 40, f"expected 40 questions, built {len(rows)}"
    assert sum(r["slice"] == "in_corpus" for r in rows) == 28
    assert sum(r["slice"] == "adversarial" for r in rows) == 12
    out = ROOT / "data" / "questions.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}: 40 questions (28 in-corpus, 12 adversarial)")


if __name__ == "__main__":
    main()
