# mlops-evidence-lab

**Question.** Can the two pieces of MLOps evidence a hiring panel
actually asks for be produced honestly on a local stack: (1) a
tracked, registered, reproducible model line whose published metrics
survive a from-scratch rerun under experiment tracking, and (2) a
standardized RAG evaluation whose failure cases are published, not
hidden?

**Headline.** The MLflow-tracked rerun reproduces credit-risk-lab's
published champion exactly, AUC 0.7810 against 0.7810 (frozen
tolerance 0.005; one deterministic seed-42 run on a fixed split, so no
interval), and the registry gate blocked both candidates it was
given; the RAG evaluation has no scores yet, because the only judge
route the protocol allows returned HTTP 429.

**Field.** ML and AI research. **Job family.** MLOps and ML
engineering.

**Verdict.** Functional. PREREGISTRATION.md is the repository's first
commit (2f5de92), made before any code or result, and has not changed
since; a test pins its SHA-256. Half 1 ran end to end (four MLflow
runs, registry, gate), tests pass in CI, and RESULTS.md regenerates
unchanged from the recorded evidence in CI. Not Evaluated: H3 and H4
are NOT RUN, and the ML profile's intervals and ablations are absent
(AUDIT.md, "Threats to validity").

## Reproduce

Checks, anywhere (no data download, no model calls):

```text
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sh scripts/fetch_rag_subject.sh && .venv/bin/python scripts/build_questions.py
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/make_results.py && git diff --exit-code RESULTS.md results/verdicts.json
```

The last line is the one-command reproduction: it rebuilds RESULTS.md
and the verdicts from the recorded artifacts in results/. The full
study pipeline is listed under "Full pipeline" below.

## Lifecycle

- Question: DONE 2026-11-08
- Proposal: DONE 2026-11-08 (PREREGISTRATION.md, commit 2f5de92)
- Baseline: DONE 2026-11-08 (subject's published figures; logistic regression rerun)
- Build: DONE 2026-11-08
- Evaluate: PARTIAL 2026-11-08 (Half 1 measured; Half 2 scoring blocked by the judge route, DECISIONS.md D4, D6)
- Report: PARTIAL 2026-11-08 (AUDIT.md, RESULTS.md)
- Review: OPEN
- Release: OPEN

The protocol was frozen in PREREGISTRATION.md (the first
content commit of this repository) before any measured run.
Verdicts are computed from the recorded artifacts by
scripts/make_results.py, never assigned by hand.

## Headline verdicts

- **H1 CONFIRMED.** The MLflow-tracked rerun of
  credit-risk-lab's champion model reproduces its published
  test metrics exactly: AUC 0.7810 against a published 0.7810
  (frozen tolerance 0.005, measured delta 0.0000), KS 0.4196
  against 0.4196, Brier 0.1355 against 0.1355. The baseline
  also reproduces exactly (logistic regression AUC 0.7518).
- **H2 INCONCLUSIVE.** The registry promotion gate did its
  blocking job in both tests it was given: the tuned challenger
  (AUC 0.7756) did not clear the frozen bar (champion + 0.003
  = 0.7840) and was blocked, and the deliberately degraded
  negative control (train labels permuted, AUC 0.4828) was
  blocked on both frozen criteria. No candidate qualified for
  promotion, so the promote direction of the gate is untested;
  the frozen verdict logic calls that INCONCLUSIVE, and this
  repository reports it that way rather than re-tuning the
  challenger until it passes.
- **H3 NOT RUN and H4 NOT RUN.** Half 2 is built end to end
  (frozen 40-question set, local generation, Ragas scorer with
  the judge and embeddings wired through the Gemini connector
  route), but after one successful smoke call the route
  returned HTTP 429 on every generateContent attempt of the
  stopped scoring run and on 5 spaced probes over about 4
  minutes (results/gemini_call_log.jsonl, DECISIONS.md D4). No Ragas
  scores exist, and none are claimed. Scoring completes with
  one command when the route answers again; see below.

Full tables and per-hypothesis text: RESULTS.md. The study
report, with what the evidence supports, incidents, and threats
to validity: AUDIT.md.

## Half 1: the model line

Subjects are pinned and read-only. credit-risk-lab at commit
fee580d0 supplies the data file (SHA-256 verified against the
subject's own manifest), the champion configuration, the
published metrics, and the leakage-safe protocol (stratified
70/30 split, seed 42, preprocessing fit on train only). This
repository re-implements that protocol (DECISIONS.md D1),
trains four candidates under MLflow on a local sqlite store,
and registers the champion rerun as credit_pd_champion
version 1 under the alias champion.

| Candidate | AUC | KS | Brier |
| --- | --- | --- | --- |
| Logistic regression | 0.7518 | 0.4008 | 0.1416 |
| Champion rerun | 0.7810 | 0.4196 | 0.1355 |
| Challenger (tuned) | 0.7756 | 0.4235 | 0.1363 |
| Negative control (labels permuted) | 0.4828 | 0.0281 | 0.1727 |

Gate rule (frozen): promote only if candidate AUC is at least
champion AUC + 0.003 and candidate Brier is at most champion
Brier + 0.002, on the same frozen test split. Both candidates
were blocked, with per-criterion reasons in
results/gate_decisions.json. Notably, the tuned challenger has
the best KS of the field (0.4235) and still fails, because the
gate reads the frozen criteria, not the most flattering one.

## Half 2: the RAG evaluation (harness complete, scoring pending)

The system under test is rag-foundation's own extractive
pipeline (pinned commit 007e3323, vendored verbatim with
provenance), run at its defaults over its own 20-document
corpus. The frozen question set (data/questions.json) holds 40
questions: 28 in-corpus with reference answers taken from the
corpus text, and 12 adversarial questions (30 percent) labelled
by design: absent details, false premises, and out-of-corpus
topics, whose correct behaviour is to say the corpus cannot
answer. Generation is local and deterministic; the samples are
recorded in results/rag_samples.json. Locally, the pipeline
already shows the failure mode the hypotheses target: it
returns non-empty extractive answers to 10 of the 12
adversarial questions (a diesel-engine question answered with
a sentence about storms), and stays correctly silent on 2.

Scoring uses Ragas (faithfulness, answer relevancy, context
precision, context recall) with the judge LLM and embeddings
reached only through the connected Gemini connector route,
wrapped for Ragas in src/ragas_wrappers.py, under the frozen
pacing and retry protocol in src/gemini_client.py. H3 freezes
the adversarial faithfulness line (below 0.50 confirms) and H4
the relevancy gap line (at least 0.30 confirms). The route's
refusal is recorded, not routed around: no other provider, no
new keys, per the protocol.

## Full pipeline

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    sh scripts/fetch_credit_data.sh          # verifies the data SHA-256
    sh scripts/fetch_rag_subject.sh          # pinned rag-foundation clone
    .venv/bin/python scripts/build_questions.py
    .venv/bin/python src/train_candidates.py # Half 1 runs + registry
    .venv/bin/python src/gate.py             # frozen gate, both candidates
    .venv/bin/python src/run_rag.py          # Half 2 local generation
    .venv/bin/python src/score_ragas.py       # Half 2 scoring (needs the route)
    .venv/bin/python scripts/make_results.py # RESULTS.md + verdicts
    .venv/bin/python -m pytest tests/        # unit tests

## Why this matters beyond this repository

A model registry and a published evaluation are not paperwork
around the real work; near money and health decisions, they
are the real work. A credit model that cannot be re-run to the
same number it was approved at is not a model, it is a rumour.
A retrieval system whose failures are trimmed from the report
will answer a question it cannot support, confidently, in
production. The proof that a small team can be trusted with
either is not a dashboard. It is an audit trail that a stranger
can re-run: a frozen protocol, recorded attempts including the
refused ones, and verdicts published whether they flatter the
author or not. This repository tries to be that trail,
including the part where the trail currently stops at a
rate-limited door and says so.

## Layout

Files named in the locked PREREGISTRATION.md keep their names; the
second column gives the role each plays in the portfolio standard.
There is no experiments/ folder: the raw evidence lives in results/
beside the generated files, as the lock names it.

| Path | Role | What it holds |
| --- | --- | --- |
| PREREGISTRATION.md | proposal | The frozen protocol and pass lines (commit 2f5de92); a test fails if one byte changes |
| AUDIT.md | report | What the evidence supports, how it is protected, incidents, threats to validity, the Jev claim check |
| RESULTS.md | results | Generated verdicts and tables (partial while Half 2 is un-run; it says so in its own text) |
| DECISIONS.md | decisions | Every deviation and incident, dated |
| results/ | evidence | Run artifacts and manifest, gate decisions, the frozen RAG samples, the Gemini call log, the Jev claim check, verdicts.json |
| data/ | evidence | The frozen 40-question set, adversarial questions, reference answers |
| src/ | code | Credit protocol, MLflow training, the gate, the vendored subject pipeline (src/ragref), the Gemini client, the Ragas wrappers and scorer |
| scripts/ | code | Data and subject fetchers, question builder, route probe, results generator, Jev claim check |
| tests/ | tests | Metric, gate and verdict logic; layout, lock and no-public-grade checks |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | release | Change history, citation, vulnerability reporting, Apache-2.0 |

## License

See LICENSE.
