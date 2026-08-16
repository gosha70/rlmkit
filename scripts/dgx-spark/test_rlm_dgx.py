"""Manual smoke test: RLMKit against Ollama and/or vLLM on DGX Spark.

Both backends are probed at startup. Tests for an unavailable backend are
skipped automatically — no need to stop the script.

Usage:
    uv run python scripts/dgx-spark/test_rlm_dgx.py [dgx-spark-ip] [ollama-model] [vllm-model]

Defaults:
    dgx-spark-ip   192.168.1.23
    ollama-model   gpt-oss:20b
    vllm-model     Qwen/Qwen2.5-7B-Instruct

Examples:
    uv run python scripts/dgx-spark/test_rlm_dgx.py 192.168.1.23
    uv run python scripts/dgx-spark/test_rlm_dgx.py 192.168.1.23 gpt-oss:20b Qwen/Qwen2.5-7B-Instruct
"""

import sys
import urllib.request
import urllib.error
from collections.abc import Callable
from typing import Any

from rlmstudio import interact
from rlmstudio.api import InteractResult

# -- Config ------------------------------------------------------------------

DGX_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.23"
OLLAMA_MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt-oss:20b"
VLLM_MODEL = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-7B-Instruct"

OLLAMA_BASE = f"http://{DGX_IP}:11434"
VLLM_BASE = f"http://{DGX_IP}:8000/v1"

SHORT = "The Eiffel Tower was built in 1889 in Paris, France."

MEDIUM = (
    "This is a report on renewable energy. "
    "Solar power accounted for 30% of new installations in 2023, "
    "wind power 40%, and hydroelectric 20%. "
    "The remaining 10% came from geothermal and biomass. "
    "Total investment reached $500B globally. "
    "Europe led adoption with 35% market share, "
    "followed by Asia-Pacific at 30%."
) * 5

# -- Helpers -----------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"


def probe(url: str, timeout: int = 5) -> bool:
    """Return True if the URL responds with any HTTP status."""
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
        return True
    except urllib.error.HTTPError:
        return True  # got a response (even 4xx) — server is up
    except Exception:
        return False


def _is_success(result: Any) -> bool:
    """Return True only when the call genuinely succeeded."""
    if isinstance(result, InteractResult):
        # raw_result carries the authoritative success flag; fall back to
        # checking for a non-empty answer when it is absent (compare mode).
        if result.raw_result is not None:
            return bool(result.raw_result.success)
        return bool(result.answer)
    return bool(result)  # type: ignore[truthy-bool]


def run(label: str, fn: Callable[[], Any]) -> Any:
    try:
        result = fn()
        status = PASS if _is_success(result) else FAIL
        print(f"  [{status}] {label}")
        return result
    except Exception as exc:
        print(f"  [{FAIL}] {label} — {type(exc).__name__}: {exc}")
        return None


def print_result(r: Any) -> None:
    if r is None:
        return
    answer = r.answer[:120] + ("..." if len(r.answer) > 120 else "")
    print(f"           answer : {answer}")
    if hasattr(r, "steps") and r.steps:
        print(f"           steps  : {r.steps}")
    print(f"           tokens : {r.total_tokens}")


# -- Backend availability check ----------------------------------------------

print(f"\nDGX Spark RLM smoke test  —  {DGX_IP}\n")

ollama_up = probe(f"{OLLAMA_BASE}/api/tags")
vllm_up = probe(f"{VLLM_BASE}/models")

print(
    f"  Ollama  ({OLLAMA_BASE})  {'UP  — model: ' + OLLAMA_MODEL if ollama_up else RED + 'NOT REACHABLE' + RESET}"
)
print(
    f"  vLLM    ({VLLM_BASE})  {'UP  — model: openai/' + VLLM_MODEL if vllm_up else RED + 'NOT REACHABLE' + RESET}"
)
print()

# -- Ollama tests ------------------------------------------------------------

print(f"{'=' * 60}")
print(f"  Ollama  ({OLLAMA_MODEL})")
print(f"{'=' * 60}")

if not ollama_up:
    print(f"  [{SKIP}] all Ollama tests — backend not reachable\n")
else:
    r = run(
        "O1  direct / SHORT",
        lambda: interact(
            SHORT,
            "When was the Eiffel Tower built?",
            mode="direct",
            provider="ollama",
            model=OLLAMA_MODEL,
            api_base=OLLAMA_BASE,
            timeout=120,
        ),
    )
    print_result(r)

    r = run(
        "O2  rlm    / MEDIUM",
        lambda: interact(
            MEDIUM,
            "Summarize the key themes",
            mode="rlm",
            provider="ollama",
            model=OLLAMA_MODEL,
            api_base=OLLAMA_BASE,
            max_steps=2,
            timeout=300,
        ),
    )
    print_result(r)

print()

# -- vLLM tests --------------------------------------------------------------

print(f"{'=' * 60}")
print(f"  vLLM  (openai/{VLLM_MODEL})")
print(f"{'=' * 60}")

if not vllm_up:
    print(f"  [{SKIP}] all vLLM tests — backend not reachable")
    print(f"          Start it on DGX Spark with:")
    print(f"            bash scripts/dgx-spark/vllm/serve-vllm-dgx.sh {VLLM_MODEL}")
    print(f"          If startup fails with a memory-profiling assertion, try:")
    print(
        f'            bash scripts/dgx-spark/vllm/serve-vllm-dgx.sh {VLLM_MODEL} 0.0.0.0 8000 "" 8192 8G\n'
    )
else:
    vllm_model_prefixed = f"openai/{VLLM_MODEL}"

    r = run(
        "V1  direct / SHORT",
        lambda: interact(
            SHORT,
            "When was the Eiffel Tower built?",
            mode="direct",
            provider="litellm",
            model=vllm_model_prefixed,
            api_base=VLLM_BASE,
            api_key="none",
            timeout=120,
            num_retries=0,  # local server — retries multiply timeout hangs
        ),
    )
    print_result(r)

    r = run(
        "V2  rlm    / MEDIUM",
        lambda: interact(
            MEDIUM,
            "Summarize the key themes",
            mode="rlm",
            provider="litellm",
            model=vllm_model_prefixed,
            api_base=VLLM_BASE,
            api_key="none",
            max_steps=2,
            timeout=300,
            num_retries=0,  # local server — retries multiply timeout hangs
        ),
    )
    print_result(r)

print()
