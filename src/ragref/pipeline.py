"""Level 2, Build: hybrid search with reranking and a clean query API.

Python standard library only. Four stages over one chunk set, every
formula in the open:

1. Chunk: the Learn approach, re-implemented compatibly. Documents
   split into word windows (default 40 words, 10 words of overlap).
   Tokens are content tokens exactly as in Learn: lowercase words and
   numbers, stopwords removed, with the same crude trailing-s stem.
2. Retrieve, two ways over the same chunks:
   - BM25: score = sum over distinct query terms of
     idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)),
     with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), k1 = 1.5,
     b = 0.75, N the number of chunks, dl the chunk length in
     content tokens, avgdl the mean chunk length.
   - Vector: TF-IDF cosine exactly as in Learn. Term frequency is
     the raw count, idf(t) = log((1 + N) / (1 + df(t))) + 1, and the
     ranking is cosine similarity between the query vector and each
     chunk vector. This is the vector side of the hybrid. It is a
     lexical sparse vector, not a dense embedding from a model, and
     no file here claims otherwise.
3. Fuse: reciprocal rank fusion over the two full rankings.
   fused(c) = sum over retrievers of 1 / (RRF_K + rank), with ranks
   1-based and RRF_K = 60. A chunk absent from a ranking gets no
   term from that retriever.
4. Rerank: the hybrid top RERANK_DEPTH candidates are re-scored by
   query coverage first (the share of distinct query content terms
   the chunk contains), then token F1 between the query term set and
   the chunk term set, then the hybrid score. This is the style
   proven in the rag-eval-harness Deep level: coverage and F1 see
   the breadth of the match that a length-normalised score can miss.
   Ties keep the earlier order, so reranking is deterministic.

Answer: extractive only, with the Learn citation discipline. From
each final chunk the sentence with the highest IDF-weighted query
coverage is picked (the Learn answer rule, using the vector IDF);
the first sentence is always kept, later sentences must reach 0.50
coverage. The threshold rises from the Learn 0.30 because this
corpus has sibling documents that share topic terms (the three
Northlight documents all carry northlight and library), so a 0.30
line lets a sibling topic sentence into the answer on the strength
of the shared names alone; the first demo run showed exactly that,
and the fix is recorded in the level README. Every returned
sentence carries one citation (doc id plus chunk id). Nothing is
generated.

All rankings drop zero scores and break ties on chunk_id, so every
stage is deterministic: the same query twice gives the same result.
"""

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or as at by "
    "it its this that these those from with without into over under "
    "how what where when who why do does did can could during which "
    "each".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")

K1 = 1.5
B = 0.75
RRF_K = 60
RERANK_DEPTH = 8
MIN_SECOND_COVERAGE = 0.50


def _stem(token):
    """The Learn stem, unchanged: drop a trailing s when the token is
    longer than 3 characters and does not end in ss or sis."""
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith("ss")
        and not token.endswith("sis")
    ):
        return token[:-1]
    return token


def content_tokens(text):
    """Lowercase word and number tokens, stopwords removed, lightly
    stemmed, order kept. Identical rule to Learn."""
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text):
    """Split text into sentences, keeping the ending punctuation."""
    parts = [p.strip() for p in SENTENCE_RE.findall(text)]
    return [p for p in parts if p]


class Chunk:
    """One chunk of one document. chunk_id looks like bees-c0."""

    def __init__(self, doc_id, index, text):
        self.doc_id = doc_id
        self.index = index
        self.chunk_id = f"{doc_id}-c{index}"
        self.text = text

    def __repr__(self):
        return f"Chunk({self.chunk_id!r})"


def chunk_document(doc_id, text, chunk_size=40, overlap=10):
    """Split one document into word windows with overlap, exactly the
    Learn rule: step is chunk_size minus overlap, the final partial
    window is kept, and overlap must be below chunk_size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and below chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append(Chunk(doc_id, index, " ".join(window)))
        index += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_corpus(documents, chunk_size=40, overlap=10):
    """Chunk every document in a list of (doc_id, text) pairs."""
    chunks = []
    for doc_id, text in documents:
        chunks.extend(chunk_document(doc_id, text, chunk_size, overlap))
    return chunks


def compute_idf(chunks_tokens):
    """Smoothed TF-IDF IDF, the Learn formula:
    idf(t) = log((1 + N) / (1 + df(t))) + 1, natural log."""
    n = len(chunks_tokens)
    df = Counter()
    for tokens in chunks_tokens:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}


def tfidf_vector(tokens, idf):
    """TF-IDF vector as a dict: raw term count times IDF. Terms
    missing from the IDF table are skipped, as in Learn."""
    counts = Counter(tokens)
    return {t: c * idf[t] for t, c in counts.items() if t in idf}


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two sparse dict vectors. Empty or
    zero vectors score 0.0."""
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class BM25Retriever:
    """BM25 over a fixed list of chunks. Formula in the module docstring."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.doc_tokens = [content_tokens(c.text) for c in self.chunks]
        self.avgdl = (
            sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )
        n = len(self.chunks)
        df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        self._tf = [Counter(tokens) for tokens in self.doc_tokens]

    def score(self, query_tokens, idx):
        """BM25 score of chunk idx for the query token list."""
        tf = self._tf[idx]
        dl = len(self.doc_tokens[idx])
        if dl == 0 or self.avgdl == 0.0:
            return 0.0
        score = 0.0
        for term in set(query_tokens):
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + K1 * (1 - B + B * dl / self.avgdl)
            score += self.idf.get(term, 0.0) * freq * (K1 + 1) / denom
        return score

    def retrieve(self, query, top_k=None):
        """Return (chunk, score) pairs, best first, zero scores
        dropped, ties on chunk_id. top_k None returns the full ranking."""
        query_tokens = content_tokens(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = self.score(query_tokens, i)
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored if top_k is None else scored[:top_k]


class VectorRetriever:
    """TF-IDF cosine retrieval over a fixed list of chunks, the Learn
    retriever carried to Build. This is the vector side of the hybrid."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        token_lists = [content_tokens(c.text) for c in self.chunks]
        self.idf = compute_idf(token_lists)
        self.vectors = [tfidf_vector(tokens, self.idf) for tokens in token_lists]

    def retrieve(self, query, top_k=None):
        """Return (chunk, score) pairs, best first, zero scores
        dropped, ties on chunk_id. top_k None returns the full ranking."""
        query_vec = tfidf_vector(content_tokens(query), self.idf)
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = cosine_similarity(query_vec, vec)
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored if top_k is None else scored[:top_k]


def rrf_fuse(rankings, k=RRF_K):
    """Reciprocal rank fusion of several rankings.

    rankings is a list of rankings, each a list of (chunk, score)
    pairs in rank order. fused(c) = sum of 1 / (k + rank) over the
    rankings containing c, ranks 1-based. Returns (chunk, fused)
    pairs, best first, ties on chunk_id.
    """
    fused = {}
    chunks_by_id = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.chunk_id] = chunk
    pairs = [(chunks_by_id[cid], score) for cid, score in fused.items()]
    pairs.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
    return pairs


def coverage_f1(query, chunk_text):
    """Return (coverage, f1) of the query content terms in a chunk.

    coverage is the share of distinct query terms the chunk contains.
    f1 is the token F1 between the query term set and the chunk term
    set: precision is shared over chunk terms, recall is coverage.
    """
    q = set(content_tokens(query))
    p = set(content_tokens(chunk_text))
    if not q or not p:
        return 0.0, 0.0
    shared = len(q & p)
    if shared == 0:
        return 0.0, 0.0
    coverage = shared / len(q)
    precision = shared / len(p)
    f1 = 2 * precision * coverage / (precision + coverage)
    return coverage, f1


def rerank(query, hybrid_hits, depth=RERANK_DEPTH):
    """Re-score the hybrid top candidates by coverage, then F1, then
    the hybrid score. hybrid_hits is a list of (chunk, hybrid_score)
    in hybrid order. Only the first depth entries are re-scored; the
    rest keep their hybrid order after them. Returns a list of dicts
    with chunk, hybrid_score, coverage, and f1, best first. Ties keep
    the earlier hybrid order, so the result is deterministic.
    """
    head = list(hybrid_hits[:depth])
    tail = list(hybrid_hits[depth:])
    decorated = []
    for idx, (chunk, score) in enumerate(head):
        coverage, f1 = coverage_f1(query, chunk.text)
        decorated.append({
            "chunk": chunk,
            "hybrid_score": score,
            "coverage": coverage,
            "f1": f1,
            "_order": idx,
        })
    decorated.sort(
        key=lambda d: (-d["coverage"], -d["f1"], -d["hybrid_score"], d["_order"])
    )
    for chunk, score in tail:
        coverage, f1 = coverage_f1(query, chunk.text)
        decorated.append({
            "chunk": chunk,
            "hybrid_score": score,
            "coverage": coverage,
            "f1": f1,
            "_order": len(decorated),
        })
    for entry in decorated:
        del entry["_order"]
    return decorated


def doc_ranking(ranked_chunks):
    """Collapse a chunk ranking to a document ranking: each doc id
    once, in the order its first chunk appears."""
    seen = []
    for chunk in ranked_chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    return seen


def evaluate_rankings(ranked_docs_per_question, expected_docs):
    """Ablation metrics over doc rankings.

    ranked_docs_per_question is a list of doc rankings (lists of doc
    ids, best first), expected_docs the matching expected doc ids.
    Returns top-1 accuracy, recall at 3 (expected doc among the first
    3 distinct docs), and MRR (mean reciprocal rank of the expected
    doc in the full doc ranking, 0 when absent).
    """
    n = len(expected_docs)
    if n == 0:
        return {"n": 0, "top1_accuracy": 0.0, "recall_at_3": 0.0, "mrr": 0.0}
    top1 = 0
    in3 = 0
    rr_sum = 0.0
    for ranking, expected in zip(ranked_docs_per_question, expected_docs):
        if ranking and ranking[0] == expected:
            top1 += 1
        if expected in ranking[:3]:
            in3 += 1
        if expected in ranking:
            rr_sum += 1.0 / (ranking.index(expected) + 1)
    return {
        "n": n,
        "top1_accuracy": top1 / n,
        "recall_at_3": in3 / n,
        "mrr": rr_sum / n,
    }


class RagPipeline:
    """Clean query API over the hybrid pipeline.

    query(question, top_k=3, max_sentences=2) returns a dict:
    - question: the question as given
    - answer: the picked sentences joined by a space
    - sentences: one entry per answer sentence with sentence, doc_id,
      chunk_id, coverage, f1, and hybrid_score
    - citations: (sentence, doc_id, chunk_id) per answer sentence, in
      the same order, exactly one per sentence
    - stages: per-stage breakdown, each a list of up to top_k entries
      with chunk_id, doc_id, and score (the reranked stage also
      carries coverage and f1)
    """

    def __init__(self, documents, chunk_size=40, overlap=10):
        pairs = [(d["id"], d["text"]) for d in documents]
        self.chunks = chunk_corpus(pairs, chunk_size, overlap)
        self.bm25 = BM25Retriever(self.chunks)
        self.vector = VectorRetriever(self.chunks)

    def rankings(self, question):
        """All four chunk rankings for a question: bm25, vector,
        hybrid (fused pairs), reranked (dicts from rerank)."""
        bm25_ranking = self.bm25.retrieve(question)
        vector_ranking = self.vector.retrieve(question)
        hybrid = rrf_fuse([bm25_ranking, vector_ranking])
        reranked = rerank(question, hybrid)
        return {
            "bm25": bm25_ranking,
            "vector": vector_ranking,
            "hybrid": hybrid,
            "reranked": reranked,
        }

    def query(self, question, top_k=3, max_sentences=2):
        stages = self.rankings(question)
        final = stages["reranked"][:top_k]
        idf = self.vector.idf
        query_terms = {t for t in content_tokens(question) if t in idf}
        query_weight = sum(idf[t] for t in query_terms)
        picked = []
        for entry in final:
            chunk = entry["chunk"]
            best_sentence = None
            best_coverage = 0.0
            for sentence in split_sentences(chunk.text):
                shared = query_terms & set(content_tokens(sentence))
                coverage = (
                    sum(idf[t] for t in shared) / query_weight
                    if query_weight else 0.0
                )
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_sentence = sentence
            if best_sentence is None:
                continue
            if picked and best_coverage < MIN_SECOND_COVERAGE:
                continue
            picked.append({
                "sentence": best_sentence,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "coverage": entry["coverage"],
                "f1": entry["f1"],
                "hybrid_score": entry["hybrid_score"],
                "sentence_coverage": best_coverage,
            })
            if len(picked) >= max_sentences:
                break

        def stage_view(pairs):
            return [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": s}
                for c, s in pairs[:top_k]
            ]

        breakdown = {
            "bm25": stage_view(stages["bm25"]),
            "vector": stage_view(stages["vector"]),
            "hybrid": stage_view(stages["hybrid"]),
            "reranked": [
                {
                    "chunk_id": e["chunk"].chunk_id,
                    "doc_id": e["chunk"].doc_id,
                    "score": e["hybrid_score"],
                    "coverage": e["coverage"],
                    "f1": e["f1"],
                }
                for e in stages["reranked"][:top_k]
            ],
        }
        return {
            "question": question,
            "answer": " ".join(p["sentence"] for p in picked),
            "sentences": picked,
            "citations": [
                (p["sentence"], p["doc_id"], p["chunk_id"]) for p in picked
            ],
            "stages": breakdown,
        }


def build_pipeline(chunk_size=40, overlap=10):
    """Build the default pipeline over the Build corpus."""
    from corpus import DOCUMENTS

    return RagPipeline(DOCUMENTS, chunk_size, overlap)
