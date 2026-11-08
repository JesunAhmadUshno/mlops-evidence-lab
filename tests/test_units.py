"""Unit tests for the study's own logic: metrics, gate rule, verdicts."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from credit_common import ks_stat, psi  # noqa: E402
from gate import decide  # noqa: E402
import make_results as mr  # noqa: E402


def test_ks_bounds():
    y = np.array([0, 0, 1, 1])
    assert ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_psi_identical_is_zero():
    x = np.linspace(0, 1, 500)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_gate_rule_edges():
    champ = {"auc": 0.7810, "ks": 0.4196, "brier": 0.1355}
    ok, _ = decide(champ, {"auc": 0.7840, "ks": 0.0, "brier": 0.1375})
    assert ok
    ok, _ = decide(champ, {"auc": 0.7839, "ks": 0.0, "brier": 0.1300})
    assert not ok
    ok, _ = decide(champ, {"auc": 0.7900, "ks": 0.0, "brier": 0.1376})
    assert not ok


def _half1(champ_auc=0.7810, chal_auc=0.7900, ctrl_auc=0.49):
    base = {"auc": champ_auc, "ks": 0.42, "brier": 0.1355}
    return {
        "champion-rerun": dict(base),
        "challenger-tuned": {"auc": chal_auc, "ks": 0.43, "brier": 0.1350},
        "negative-control": {"auc": ctrl_auc, "ks": 0.03, "brier": 0.17},
        "logistic-regression": {"auc": 0.7518, "ks": 0.4008, "brier": 0.1416},
    }


def _decisions(chal="promote", ctrl="block"):
    return [
        {"candidate": "challenger-tuned", "decision": chal},
        {"candidate": "negative-control", "decision": ctrl},
    ]


def test_h2_confirmed_path():
    verdict, clears = mr.verdict_h2(_half1(), _decisions())
    assert verdict == "CONFIRMED" and clears


def test_h2_refuted_when_control_promoted():
    verdict, _ = mr.verdict_h2(_half1(), _decisions(ctrl="promote"))
    assert verdict == "REFUTED"


def test_h2_inconclusive_when_challenger_below_bar():
    verdict, clears = mr.verdict_h2(_half1(chal_auc=0.7756),
                                    _decisions(chal="block"))
    assert verdict == "INCONCLUSIVE" and not clears


def test_h1_tolerance():
    verdict, delta = mr.verdict_h1(_half1())
    assert verdict == "CONFIRMED" and delta == 0.0
    verdict, _ = mr.verdict_h1(_half1(champ_auc=0.7700))
    assert verdict == "REFUTED"


def _rows(slice_name, metric_values, key="faithfulness", empty=0):
    rows = []
    for i, v in enumerate(metric_values):
        rows.append({"id": f"{slice_name}-{i}", "slice": slice_name,
                     "answer_empty": False, "faithfulness": v,
                     "answer_relevancy": v})
    for j in range(empty):
        rows.append({"id": f"{slice_name}-e{j}", "slice": slice_name,
                     "answer_empty": True, "faithfulness": None,
                     "answer_relevancy": None})
    return rows


def test_h3_silent_rows_do_not_count_as_scored():
    rows = _rows("adversarial", [0.9] * 10, empty=2)
    verdict, f_adv, scored, silent = mr.verdict_h3(rows)
    assert (verdict, scored, silent) == ("REFUTED", 10, 2)
    assert f_adv == pytest.approx(0.9)


def test_h3_inconclusive_below_nine_scored():
    rows = _rows("adversarial", [0.1] * 8, empty=4)
    verdict, _, scored, _ = mr.verdict_h3(rows)
    assert verdict == "INCONCLUSIVE" and scored == 8


def test_h4_gap_rule():
    rows = _rows("in_corpus", [0.9] * 28, key="answer_relevancy")
    rows += _rows("adversarial", [0.4] * 12, key="answer_relevancy")
    for r in rows:
        r["answer_relevancy"] = 0.9 if r["slice"] == "in_corpus" else 0.4
    verdict, r_in, r_adv, gap = mr.verdict_h4(rows)
    assert verdict == "CONFIRMED" and gap == pytest.approx(0.5)


def test_question_set_shape():
    questions = json.loads((ROOT / "data" / "questions.json").read_text())
    assert len(questions) == 40
    assert sum(q["slice"] == "in_corpus" for q in questions) == 28
    assert sum(q["slice"] == "adversarial" for q in questions) == 12
    adv_types = [q["category"] for q in questions if q["slice"] == "adversarial"]
    assert adv_types.count("absent_detail") == 4
    assert adv_types.count("false_premise") == 4
    assert adv_types.count("out_of_corpus") == 4
    for q in questions:
        if q["slice"] == "adversarial":
            assert q["reference"] == "Not answerable from the corpus."
