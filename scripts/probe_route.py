"""One single-attempt probe of the Gemini route (no retries).

Used only to gate the start of the scoring run (DECISIONS.md
D4): the scoring protocol itself always uses the frozen pacing
and retry ladder in src/gemini_client.py. Prints the HTTP
status and exits 0 on 200, 1 otherwise.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamic_credentials import add_surrogate_to_request, read_json_response  # noqa: E402

import gemini_client  # noqa: E402


def main() -> int:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{gemini_client.JUDGE_MODEL}:generateContent")
    body = json.dumps({
        "contents": [{"parts": [{"text": "Reply with exactly the word: probe"}]}],
        "generationConfig": {"temperature": 0},
    }).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    add_surrogate_to_request(request, "custom.gemini",
                             allowed_hosts=["generativelanguage.googleapis.com"])
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = read_json_response(response)
        text = "".join(
            p.get("text", "")
            for p in data.get("candidates", [{}])[0]
            .get("content", {}).get("parts", []))
        print(f"200: {text.strip()[:40]}")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"{exc.code}")
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
