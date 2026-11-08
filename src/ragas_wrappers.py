"""Ragas LLM and embedding interfaces backed by the Gemini route.

Both wrappers call src/gemini_client.py, which implements the
frozen access protocol (connector credential, pacing, retries,
call logging). No other provider is reachable from here.
"""
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ragas_compat  # noqa: E402

ragas_compat.apply()

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import BaseRagasLLM  # noqa: E402

import gemini_client  # noqa: E402


@dataclass
class GeminiLLM(BaseRagasLLM):
    """BaseRagasLLM over the connector's Gemini generateContent route."""

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None,
                      callbacks=None) -> LLMResult:
        text = prompt.to_string()
        generations = []
        for _ in range(n):
            answer = gemini_client.generate_text(
                text, temperature=temperature if temperature is not None else 0.0)
            generations.append([Generation(text=answer)])
        return LLMResult(generations=generations)

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None,
                             callbacks=None) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class GeminiEmbeddings(BaseRagasEmbeddings):
    """BaseRagasEmbeddings over the connector's Gemini embedding route."""

    def embed_query(self, text: str) -> List[float]:
        return gemini_client.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return gemini_client.embed_batch(list(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
