"""A paced, retrying Gemini client over the connected connector route.

Implements exactly the frozen access protocol of
PREREGISTRATION.md Half 2: requests go only to
generativelanguage.googleapis.com with the credential attached
by the connector's dynamic-credentials helper under the stored
name custom.gemini; at least 4.0 seconds separate successive
requests; HTTP 429 and 5xx responses are retried up to 4 times
with backoff of 15, 30, 60, 120 seconds; every attempt is
logged to results/gemini_call_log.jsonl.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOST = "generativelanguage.googleapis.com"
BASE = f"https://{HOST}/v1beta/models"
JUDGE_MODEL = "gemini-3-flash-preview"
EMBED_MODEL = "gemini-embedding-001"
MIN_INTERVAL = 4.0
BACKOFF = [15, 30, 60, 120]
LOG_PATH = ROOT / "results" / "gemini_call_log.jsonl"


class GeminiError(RuntimeError):
    pass


class Pacer:
    """One global limiter: at least MIN_INTERVAL between requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + MIN_INTERVAL - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_PACER = Pacer()


def _log(kind: str, model: str, status, attempt: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind, "model": model, "status": status,
            "attempt": attempt,
        }) + "\n")


def _post(url: str, body: dict, kind: str, model: str) -> dict:
    payload = json.dumps(body).encode()
    last_error = None
    for attempt in range(len(BACKOFF) + 1):
        _PACER.wait()
        request = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"})
        add_surrogate_to_request(request, "custom.gemini", allowed_hosts=[HOST])
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                _log(kind, model, 200, attempt)
                return read_json_response(response)
        except urllib.error.HTTPError as exc:
            _log(kind, model, exc.code, attempt)
            last_error = exc
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt < len(BACKOFF):
                    time.sleep(BACKOFF[attempt])
                    continue
            raise GeminiError(f"{kind} HTTP {exc.code} after {attempt} retries") from exc
        except Exception as exc:  # network errors: same retry class as 5xx
            _log(kind, model, f"error:{type(exc).__name__}", attempt)
            last_error = exc
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt])
                continue
            raise GeminiError(f"{kind} failed after {attempt} retries: {exc}") from exc
    raise GeminiError(f"{kind} failed: {last_error}")


def generate_text(prompt: str, temperature: float = 0.0) -> str:
    url = f"{BASE}/{JUDGE_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    data = _post(url, body, "generate", JUDGE_MODEL)
    parts = []
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in part:
            parts.append(part["text"])
    text = "".join(parts).strip()
    if not text:
        raise GeminiError("generate returned no text")
    return text


def embed(text: str) -> list[float]:
    url = f"{BASE}/{EMBED_MODEL}:embedContent"
    body = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
    }
    data = _post(url, body, "embed", EMBED_MODEL)
    return data["embedding"]["values"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    url = f"{BASE}/{EMBED_MODEL}:batchEmbedContents"
    out: list[list[float]] = []
    for start in range(0, len(texts), 50):
        chunk = texts[start:start + 50]
        body = {"requests": [
            {"model": f"models/{EMBED_MODEL}",
             "content": {"parts": [{"text": t}]}} for t in chunk
        ]}
        data = _post(url, body, "batch_embed", EMBED_MODEL)
        out.extend(e["values"] for e in data["embeddings"])
    return out
