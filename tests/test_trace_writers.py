"""AC-29 — CI guard enforcing assistant-role trace writer completeness.

AST-walks ``src/rlmstudio/application/use_cases/`` for every dict literal
that contains ``TRACE_KEY_ROLE: "assistant"`` and asserts each one also
populates the four prefill/decode telemetry keys (``ttft_ms``,
``decode_ms``, ``cached_tokens``, ``cache_write_tokens``), OR carries
a nearby ``# telemetry-exempt: <reason>`` comment.

The AST approach (vs. a regex over source text) is resilient to
nested dict literals and future formatting changes — a writer that
happens to contain a nested ``{"note": {...}}`` wouldn't fool a
balanced-brace parser the way it could fool a regex with ``[^{}]``.
"""

from __future__ import annotations

import ast
import pathlib

_USE_CASES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "rlmstudio"
    / "application"
    / "use_cases"
)

_REQUIRED_KEYS = frozenset({"ttft_ms", "decode_ms", "cached_tokens", "cache_write_tokens"})
_ASSISTANT_ROLE_KEY = "TRACE_KEY_ROLE"
_ASSISTANT_ROLE_VALUE = "assistant"
_EXEMPT_TOKEN = "telemetry-exempt:"


def _dict_key_name(node: ast.expr) -> str | None:
    """Return the textual name of a dict key node, or None if not recognized.

    Handles the two key shapes the use-case writers use:

    - ``TRACE_KEY_ROLE: "assistant"`` — key is a :class:`ast.Name`.
    - ``"ttft_ms": value`` — key is a :class:`ast.Constant[str]`.

    Unknown key shapes (f-strings, computed keys) return ``None``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_assistant_role_pair(key_node: ast.expr, value_node: ast.expr) -> bool:
    """True when a dict key/value pair is ``TRACE_KEY_ROLE: "assistant"``."""
    if _dict_key_name(key_node) != _ASSISTANT_ROLE_KEY:
        return False
    return isinstance(value_node, ast.Constant) and value_node.value == _ASSISTANT_ROLE_VALUE


def _populated_key_names(node: ast.Dict) -> set[str]:
    """Return the set of textual key names present in the dict literal."""
    names: set[str] = set()
    for key in node.keys:
        if key is None:
            # ``**kwargs`` spread — ignore (use-case writers don't use it).
            continue
        name = _dict_key_name(key)
        if name is not None:
            names.add(name)
    return names


def _assistant_role_dicts(tree: ast.AST) -> list[ast.Dict]:
    """Find every dict node that carries ``TRACE_KEY_ROLE: "assistant"``."""
    found: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if key is None:
                continue
            if _is_assistant_role_pair(key, value):
                found.append(node)
                break
    return found


def _has_nearby_exempt_comment(source_lines: list[str], line_no: int, radius: int = 3) -> bool:
    """True when a ``# telemetry-exempt:`` comment sits within ``radius`` lines.

    Intentionally generous radius — a writer annotated a few lines
    above (e.g. on the `trace.append(` call site) still counts. The
    rule is "telemetry-exempt is explicit and nearby", not "on the
    exact same line."
    """
    start = max(0, line_no - radius - 1)
    end = min(len(source_lines), line_no + radius)
    for line in source_lines[start:end]:
        if _EXEMPT_TOKEN in line:
            return True
    return False


def test_every_assistant_site_populates_new_keys() -> None:
    """Every ``TRACE_KEY_ROLE: "assistant"`` dict literal in
    ``src/rlmstudio/application/use_cases/`` either carries the four
    prefill/decode telemetry keys OR an adjacent
    ``# telemetry-exempt: <reason>`` comment.
    """
    violations: list[str] = []

    for path in sorted(_USE_CASES_DIR.glob("*.py")):
        source = path.read_text()
        source_lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:  # pragma: no cover — would surface as a fail
            violations.append(f"{path}: failed to parse ({exc})")
            continue

        for dict_node in _assistant_role_dicts(tree):
            populated = _populated_key_names(dict_node)
            missing = sorted(_REQUIRED_KEYS - populated)
            if not missing:
                continue
            line_no = dict_node.lineno
            if _has_nearby_exempt_comment(source_lines, line_no):
                continue
            rel = path.relative_to(_USE_CASES_DIR.parent.parent.parent.parent)
            violations.append(f"{rel}:{line_no} missing {missing}")

    assert not violations, (
        "Assistant-role trace writers missing prefill/decode telemetry keys.\n"
        'Add `"ttft_ms"`, `"decode_ms"`, `"cached_tokens"`, and '
        '`"cache_write_tokens"` from the returning LLMResponseDTO, OR '
        "add a `# telemetry-exempt: <reason>` comment within ±3 lines "
        "of the dict literal.\n\n" + "\n".join(violations)
    )
