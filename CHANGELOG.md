# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning; no release is tagged yet.

## [Unreleased]

### Changed
- 2026-11-08: Narrowed the HTTP 429 account in README, AUDIT.md and
  RESULTS.md to what results/gemini_call_log.jsonl shows (one smoke
  call succeeded; the probes spanned about 4 minutes, not an hour)
  and stated that the two D6 probes were not logged. Verdicts
  unchanged.
- 2026-11-08: Adopted the portfolio standard v2 (DECISIONS.md D7).
  README opens with the question, headline, field, job family and
  verdict. AUDIT.md is the report; its grade moved out of the public
  repo (grades are internal) and its limitations now sit under
  "Threats to validity". A test fails if PREREGISTRATION.md changes by
  one byte from the lock commit 2f5de92. CI checks that RESULTS.md
  and results/verdicts.json regenerate unchanged, adds a Python 3.11
  leg for the dependency-free checks, and runs with read-only
  permissions. CodeQL and Dependabot added.

### Added
- 2026-11-08: DECISIONS.md D6: the judge route trickles single calls;
  scoring stays gated.
- 2026-11-08: README, AUDIT.md and the Jev claim check of the headline
  verdicts.
- 2026-11-08: Partial RESULTS.md: H1 CONFIRMED, H2 INCONCLUSIVE, H3
  and H4 NOT RUN (judge route returned HTTP 429, DECISIONS.md D4).
- 2026-11-08: Half 1 runs under MLflow, the registry gate decisions,
  and the Half 2 local samples.
- 2026-11-08: Harness: MLflow candidates and gate, Ragas wrappers over
  the Gemini connector, frozen question set.
- 2026-11-08: Pre-registration (2f5de92), before any code or result.
