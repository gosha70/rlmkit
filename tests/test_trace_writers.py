"""AC-29 — CI guard enforcing assistant-role trace writer completeness.

Greps ``src/rlmkit/application/use_cases/`` for every dict literal
containing ``TRACE_KEY_ROLE: "assistant"`` and asserts each one also
populates the four prefill/decode telemetry keys (``ttft_ms``,
``decode_ms``, ``cached_tokens``, ``cache_write_tokens``), OR carries
a nearby ``# telemetry-exempt: <reason>`` comment.

This blocks future PRs that add a new assistant-role writer without
attending to telemetry — per spec v1.7 §3 / AC-29.
"""

from __future__ import annotations

import pathlib
import re

_USE_CASES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "rlmkit" / "application" / "use_cases"
)

_REQUIRED_KEYS = ('"ttft_ms"', '"decode_ms"', '"cached_tokens"', '"cache_write_tokens"')

_ASSISTANT_PATTERN = re.compile(
    r'\{([^{}]*TRACE_KEY_ROLE[^{}]*"assistant"[^{}]*)\}',
    re.DOTALL,
)

_EXEMPT_COMMENT = re.compile(r"#\s*telemetry-exempt:", re.IGNORECASE)


def _line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def _nearby_lines(text: str, offset: int, radius: int = 3) -> str:
    """Return the ±radius lines around `offset` for exemption lookups."""
    line = _line_number(text, offset)
    start = max(0, line - radius - 1)
    end = line + radius
    return "\n".join(text.splitlines()[start:end])


def test_every_assistant_site_populates_new_keys() -> None:
    """Every TRACE_KEY_ROLE:"assistant" dict in use_cases/ either carries
    the four new telemetry keys or an adjacent `telemetry-exempt` comment.
    """
    violations: list[str] = []
    for path in sorted(_USE_CASES_DIR.glob("*.py")):
        text = path.read_text()
        for match in _ASSISTANT_PATTERN.finditer(text):
            block = match.group(0)
            missing = [k for k in _REQUIRED_KEYS if k not in block]
            if not missing:
                continue
            # Allow explicit exemption via nearby comment.
            nearby = _nearby_lines(text, match.start(), radius=3)
            if _EXEMPT_COMMENT.search(nearby):
                continue
            line_no = _line_number(text, match.start())
            violations.append(
                f"{path.relative_to(_USE_CASES_DIR.parent.parent.parent.parent)}:"
                f"{line_no} missing {missing}"
            )

    assert not violations, (
        "Assistant-role trace writers missing prefill/decode telemetry keys.\n"
        'Add `"ttft_ms"`, `"decode_ms"`, `"cached_tokens"`, and '
        '`"cache_write_tokens"` from the returning LLMResponseDTO, OR '
        "add a `# telemetry-exempt: <reason>` comment on the same or "
        "preceding line.\n\n" + "\n".join(violations)
    )
