"""A small client for the local Ollama judge (DECISIONS.md D9).

Talks to an Ollama server over an SSH local-forward tunnel to
the R610 (default base http://127.0.0.1:11435, forwarding to the
R610's localhost:11434). Judge model glm4:9b, embedding model
nomic-embed-text, both named in D9. Transient connection errors
(a tunnel reconnect gap) are retried up to 5 times with backoff
of 5, 10, 20, 40, 60 seconds; every completed call is logged to
results/ollama_call_log.jsonl. No pacing is imposed: the server
is local and dedicated.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11435")
JUDGE_MODEL = "glm4:9b"
EMBED_MODEL = "nomic-embed-text"
BACKOFF = [5, 10, 20, 40, 60]
LOG_PATH = ROOT / "results" / "ollama_call_log.jsonl"


class OllamaError(RuntimeError):
    pass


def _log(kind: str, model: str, status, attempt: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind, "model": model, "status": status,
            "attempt": attempt,
        }) + "\n")


def _post(path: str, body: dict, kind: str, model: str) -> dict:
    payload = json.dumps(body).encode()
    last_error = None
    for attempt in range(len(BACKOFF) + 1):
        request = urllib.request.Request(
            BASE + path, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                data = json.loads(response.read())
            _log(kind, model, 200, attempt)
            return data
        except Exception as exc:  # connection gaps and 5xx alike
            last_error = exc
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else f"error:{type(exc).__name__}"
            _log(kind, model, status, attempt)
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt])
    raise OllamaError(f"Ollama {kind} failed after retries: {last_error}")


def generate_text(prompt: str, temperature: float = 0.0) -> str:
    data = _post("/api/generate", {
        "model": JUDGE_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }, "generate", JUDGE_MODEL)
    return data.get("response", "")


def embed(text: str) -> list[float]:
    data = _post("/api/embed", {
        "model": EMBED_MODEL, "input": text,
    }, "embed", EMBED_MODEL)
    return data["embeddings"][0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    data = _post("/api/embed", {
        "model": EMBED_MODEL, "input": list(texts),
    }, "embed_batch", EMBED_MODEL)
    return data["embeddings"]
