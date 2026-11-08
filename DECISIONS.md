# Decisions

## D1: the rerun re-implements the subject pipeline

Half 1 re-implements credit-risk-lab's champion training in
src/credit_common.py from its published source at the pinned
commit, changing only the tracking layer. The alternative,
importing the subject repo as a package, would couple this lab
to a repo whose own runs write into its results directory; the
rerun must not touch the subject. The re-implementation is
line-faithful for loading, feature construction, preprocessing,
split, and model configurations, and H1 is exactly the test of
whether that faithfulness reproduces the published number.

## D2: ragas 0.4.3 needs a one-module compatibility shim

ragas 0.4.3 imports ChatVertexAI from
langchain_community.chat_models.vertexai, which
langchain-community 0.4 removed (the class moved to
langchain-google-vertexai). Downgrading the langchain stack to
the 0.3 line conflicts with ragas 0.4.3's own requirements.
src/ragas_compat.py registers the class from its new home
under the old module path before ragas is imported. ragas uses
the class only in an isinstance dispatch list; no Vertex AI
route is used anywhere in this lab. Pinned versions are in
requirements.txt.

## D3: the question set's paraphrase share

The subject's deep set has 10 paraphrase questions; Half 2
takes 4 of them (p01, p02, p05, p06) beside all 24 regression
questions, keeping the in-corpus slice at 28 so the adversarial
slice is exactly 30 percent of the 40-question set, as
registered. The 4 chosen paraphrases cover four distinct
expected documents (water-cycle, volcanoes, northlight-events,
photosynthesis).
