"""Portfolio standard checks: the lock holds, required files exist, no public grade.

Standard library only, so it runs on every CI interpreter.
"""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SHA-256 of PREREGISTRATION.md as committed in 2f5de92 (the first commit), before any code or result.
LOCKED_SHA256 = "9b0d5ec9fde660ea0c84287bc9054d685f62db7d47c1503d6b6da07f5698e1f1"
GRADE = re.compile(r"of 10,000|/ ?10,?000|Grade:|self-grade", re.IGNORECASE)


def test_preregistration_is_unchanged_since_the_lock():
    with open(os.path.join(ROOT, "PREREGISTRATION.md"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == LOCKED_SHA256, \
            "PREREGISTRATION.md changed after its lock; record an amendment in DECISIONS.md instead"


def test_required_files_exist():
    need = ["README.md", "PREREGISTRATION.md", "AUDIT.md", "RESULTS.md", "DECISIONS.md", "CHANGELOG.md",
            "CITATION.cff", "SECURITY.md", "LICENSE", "results/half1_metrics.json",
            "results/gate_decisions.json", "results/half1_manifest.json", "results/gemini_call_log.jsonl"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, "missing: " + ", ".join(missing)


def test_no_public_grade():
    for name in ("README.md", "AUDIT.md", "RESULTS.md", "DECISIONS.md", "CHANGELOG.md"):
        with open(os.path.join(ROOT, name)) as f:
            hit = GRADE.search(f.read())
        assert not hit, f"{name} carries a grade ({hit.group(0)!r}); grades are internal"


def test_grade_pattern_catches_the_old_form():
    assert GRADE.search("AUDIT.md (5,750 of 10,000)")
    assert GRADE.search("## Grade: 5,750")
