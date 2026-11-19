"""Ragas LLM and embedding interfaces backed by the local judge.

Registered under DECISIONS.md D9: same BaseRagasLLM and
BaseRagasEmbeddings shapes as src/ragas_wrappers.py, backed by
src/ollama_client.py (glm4:9b judge, nomic-embed-text
embeddings, Ollama on the R610). Verdicts produced through
these wrappers are LOCAL-JUDGE verdicts, never Gemini results.
"""
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ragas_compat  # noqa: E402

ragas_compat.apply()

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import BaseRagasLLM  # noqa: E402

import ollama_client  # noqa: E402


@dataclass
class LocalLLM(BaseRagasLLM):
    """BaseRagasLLM over the local Ollama judge (D9)."""

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None,
                      callbacks=None) -> LLMResult:
        text = prompt.to_string()
        generations = []
        for _ in range(n):
            answer = ollama_client.generate_text(
                text, temperature=temperature if temperature is not None else 0.0)
            generations.append([Generation(text=answer)])
        return LLMResult(generations=generations)

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None,
                             callbacks=None) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class LocalEmbeddings(BaseRagasEmbeddings):
    """BaseRagasEmbeddings over the local Ollama embedder (D9)."""

    def embed_query(self, text: str) -> List[float]:
        return ollama_client.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return ollama_client.embed_batch(list(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
