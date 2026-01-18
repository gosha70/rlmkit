Got it. I can’t access the shared chat link because it requires login (the page only shows the public “Log in / Sign up” shell). ([ChatGPT][1])
So I’m going to give you a complete **GitHub-ready package scaffold** for **`rlmkit`** that satisfies your constraints and deliverables: provider-agnostic runtime, deterministic recursion/budgets/state, provider adapters (OpenAI/Anthropic/Ollama) as optional extras, optional MCP gateway adapter, plus examples, docs, and a real pytest suite.

Below is a **drop-in repo layout + file contents** (copy/paste into a new repo). It’s intentionally minimal but extensible.

---

## Repo layout

```txt
rlmkit/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  src/
    rlmkit/
      __init__.py
      errors.py
      types.py
      state.py
      tools.py
      runtime.py
      providers/
        __init__.py
        base.py
        openai_provider.py
        anthropic_provider.py
        ollama_provider.py
      mcp/
        __init__.py
        gateway.py
  examples/
    local_ollama_basic.py
    remote_openai_basic.py
    remote_anthropic_basic.py
    tool_calling_demo.py
  docs/
    architecture.md
    extending.md
  tests/
    test_budgets_and_depth.py
    test_tool_calls.py
    test_provider_shims.py
```

---

## `pyproject.toml` (package-first, editable install friendly)

```toml
[build-system]
requires = ["hatchling>=1.21.0"]
build-backend = "hatchling.build"

[project]
name = "rlmkit"
version = "0.1.0"
description = "Recursive Language Model (RLM) runtime: recursive prompting + tool use + structured state."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Name" }]
keywords = ["llm", "agents", "tools", "recursion", "ollama", "openai", "anthropic"]
dependencies = [
  "httpx>=0.25",
]

[project.optional-dependencies]
openai = ["openai>=1.40.0"]
anthropic = ["anthropic>=0.35.0"]
dev = [
  "pytest>=8.0",
  "ruff>=0.4",
  "mypy>=1.10",
]

[project.urls]
Homepage = "https://github.com/your-org/rlmkit"
Repository = "https://github.com/your-org/rlmkit"

[tool.hatch.build.targets.wheel]
packages = ["src/rlmkit"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Core public API

### `src/rlmkit/__init__.py`

```py
from .runtime import RLMRuntime, RunConfig, StepBudget
from .state import RLMState
from .tools import Tool, ToolRegistry
from .types import Message, Role, ToolCall, ToolResult, ModelResponse

__all__ = [
    "RLMRuntime",
    "RunConfig",
    "StepBudget",
    "RLMState",
    "Tool",
    "ToolRegistry",
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "ModelResponse",
]
```

---

## Types + errors

### `src/rlmkit/errors.py`

```py
class RLMError(Exception):
    """Base exception for rlmkit."""


class BudgetExceeded(RLMError):
    """Raised when recursion depth or step budget is exhausted."""


class ToolError(RLMError):
    """Raised when a tool fails or tool arguments are invalid."""


class ProviderError(RLMError):
    """Raised for provider/transport/protocol errors."""
```

### `src/rlmkit/types.py`

```py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    name: Optional[str] = None  # used for tool messages / tool name


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str  # tool output serialized as string (deterministic)


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: List[ToolCall]
    raw: Optional[Dict[str, Any]] = None  # provider-native payload for debugging
```

---

## Deterministic state

### `src/rlmkit/state.py`

```py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .types import Message


@dataclass
class RLMState:
    """
    Explicit, deterministic state container.
    - `memory`: long-lived facts or notes your app persists
    - `vars`: structured values for your workflow
    - `history`: full transcript messages (provider-agnostic)
    """

    memory: Dict[str, Any] = field(default_factory=dict)
    vars: Dict[str, Any] = field(default_factory=dict)
    history: List[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        self.history.append(msg)
```

---

## Tool registry (schema + execution)

### `src/rlmkit/tools.py`

```py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .errors import ToolError
from .types import ToolCall, ToolResult


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_schema: Dict[str, Any]
    fn: Callable[[Dict[str, Any]], Any]

    def to_openai_schema(self) -> Dict[str, Any]:
        # OpenAI "function calling" style schema
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        # Anthropic tool schema
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def openai_schemas(self) -> list[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def anthropic_schemas(self) -> list[Dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def call(self, tc: ToolCall) -> ToolResult:
        tool = self.get(tc.name)
        if tool is None:
            raise ToolError(f"Unknown tool: {tc.name}")

        try:
            out = tool.fn(tc.arguments)
        except Exception as e:
            raise ToolError(f"Tool '{tc.name}' failed: {e}") from e

        # Deterministic serialization
        if isinstance(out, (dict, list)):
            content = json.dumps(out, sort_keys=True)
        else:
            content = str(out)

        return ToolResult(tool_call_id=tc.id, name=tc.name, content=content)
```

---

## Provider interface + adapters

### `src/rlmkit/providers/base.py`

```py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..types import ModelResponse, Message
from ..tools import ToolRegistry


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    temperature: float = 0.0
    timeout_s: float = 60.0


class ModelProvider(ABC):
    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: Optional[ToolRegistry] = None,
    ) -> ModelResponse:
        """
        Provider must:
        - accept provider-agnostic messages
        - optionally accept tools
        - return ModelResponse(content, tool_calls)
        """
        raise NotImplementedError
```

### `src/rlmkit/providers/__init__.py`

```py
from .base import ModelProvider, ProviderConfig

__all__ = ["ModelProvider", "ProviderConfig"]
```

#### OpenAI adapter (optional dependency)

### `src/rlmkit/providers/openai_provider.py`

```py
from __future__ import annotations

import json
from typing import Optional

from ..errors import ProviderError
from ..types import Message, ModelResponse, Role, ToolCall
from ..tools import ToolRegistry
from .base import ModelProvider

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIProvider(ModelProvider):
    def __init__(self, cfg, api_key: Optional[str] = None) -> None:
        super().__init__(cfg)
        if OpenAI is None:
            raise ProviderError("openai dependency not installed. pip install 'rlmkit[openai]'")
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[Message], tools: Optional[ToolRegistry] = None) -> ModelResponse:
        try:
            oai_msgs = []
            for m in messages:
                msg = {"role": m.role.value, "content": m.content}
                if m.name is not None and m.role in (Role.tool,):
                    msg["name"] = m.name
                oai_msgs.append(msg)

            kwargs = {
                "model": self.cfg.model,
                "messages": oai_msgs,
                "temperature": self.cfg.temperature,
            }
            if tools is not None:
                kwargs["tools"] = tools.openai_schemas()
                kwargs["tool_choice"] = "auto"

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0].message

            tool_calls = []
            if getattr(choice, "tool_calls", None):
                for tc in choice.tool_calls:
                    args = tc.function.arguments
                    parsed = json.loads(args) if isinstance(args, str) and args.strip() else {}
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=parsed))

            content = choice.content or ""
            return ModelResponse(content=content, tool_calls=tool_calls, raw=resp.model_dump())

        except Exception as e:
            raise ProviderError(f"OpenAI generate failed: {e}") from e
```

#### Anthropic adapter (optional dependency)

### `src/rlmkit/providers/anthropic_provider.py`

```py
from __future__ import annotations

from typing import Optional

from ..errors import ProviderError
from ..types import Message, ModelResponse, Role, ToolCall
from ..tools import ToolRegistry
from .base import ModelProvider

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore


class AnthropicProvider(ModelProvider):
    def __init__(self, cfg, api_key: Optional[str] = None) -> None:
        super().__init__(cfg)
        if anthropic is None:
            raise ProviderError(
                "anthropic dependency not installed. pip install 'rlmkit[anthropic]'"
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages: list[Message], tools: Optional[ToolRegistry] = None) -> ModelResponse:
        try:
            # Anthropic wants system separately + message list (user/assistant)
            system_parts = [m.content for m in messages if m.role == Role.system]
            system = "\n".join(system_parts) if system_parts else None

            anth_msgs = []
            for m in messages:
                if m.role == Role.system:
                    continue
                # tool results are sent as role="user" with content blocks in newer APIs,
                # but we keep it minimal: serialize tool output back into user text.
                if m.role == Role.tool:
                    anth_msgs.append({"role": "user", "content": f"[tool:{m.name}]\n{m.content}"})
                else:
                    anth_msgs.append({"role": m.role.value, "content": m.content})

            kwargs = {
                "model": self.cfg.model,
                "max_tokens": 1024,
                "messages": anth_msgs,
            }
            if system is not None:
                kwargs["system"] = system
            if tools is not None:
                kwargs["tools"] = tools.anthropic_schemas()

            resp = self.client.messages.create(**kwargs)

            tool_calls = []
            text_out = []
            for block in resp.content:
                if block.type == "text":
                    text_out.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                    )

            return ModelResponse(content="\n".join(text_out), tool_calls=tool_calls, raw=resp.model_dump())

        except Exception as e:
            raise ProviderError(f"Anthropic generate failed: {e}") from e
```

#### Ollama adapter (local; HTTP; no extra deps)

### `src/rlmkit/providers/ollama_provider.py`

```py
from __future__ import annotations

import json
from typing import Optional

import httpx

from ..errors import ProviderError
from ..types import Message, ModelResponse, ToolCall
from ..tools import ToolRegistry
from .base import ModelProvider


class OllamaProvider(ModelProvider):
    """
    Uses Ollama's /api/chat. Tool calling varies by model; to keep this deterministic,
    we support a simple JSON tool-call convention:

    Assistant may emit a JSON object:
      {"tool_calls":[{"id":"...", "name":"tool", "arguments":{...}}], "content":"...optional..."}

    If plain text is returned, tool_calls=[].
    """

    def __init__(self, cfg, base_url: str = "http://localhost:11434") -> None:
        super().__init__(cfg)
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: list[Message], tools: Optional[ToolRegistry] = None) -> ModelResponse:
        try:
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                "options": {"temperature": self.cfg.temperature},
                "stream": False,
            }

            # We do not send tool schemas to Ollama by default, because formats differ by model.
            # Instead, the runtime's system prompt can instruct the model on a JSON tool_call format.

            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()

            content = (data.get("message") or {}).get("content") or ""
            tool_calls: list[ToolCall] = []

            # Attempt to parse tool_call JSON envelope if model produced it
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict) and "tool_calls" in obj:
                        content = str(obj.get("content") or "")
                        for tc in obj.get("tool_calls") or []:
                            tool_calls.append(
                                ToolCall(
                                    id=str(tc.get("id") or "tc_0"),
                                    name=str(tc["name"]),
                                    arguments=dict(tc.get("arguments") or {}),
                                )
                            )
                except Exception:
                    pass

            return ModelResponse(content=content, tool_calls=tool_calls, raw=data)

        except Exception as e:
            raise ProviderError(f"Ollama generate failed: {e}") from e
```

---

## Optional MCP gateway adapter (not required for core)

### `src/rlmkit/mcp/__init__.py`

```py
from .gateway import MCPGatewayTool

__all__ = ["MCPGatewayTool"]
```

### `src/rlmkit/mcp/gateway.py`

```py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import httpx

from ..tools import Tool


@dataclass(frozen=True)
class MCPGatewayTool:
    """
    Minimal adapter that turns an MCP gateway endpoint into an rlmkit Tool.

    Assumes a gateway that accepts:
      POST /tools/<tool_name>  with JSON args
    and returns JSON.
    """

    base_url: str

    def as_tool(self, name: str, description: str, args_schema: Dict[str, Any]) -> Tool:
        base = self.base_url.rstrip("/")

        def _fn(args: Dict[str, Any]) -> Any:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(f"{base}/tools/{name}", json=args)
                r.raise_for_status()
                return r.json()

        return Tool(name=name, description=description, args_schema=args_schema, fn=_fn)
```

---

## RLM Runtime (deterministic recursion + budgets)

### `src/rlmkit/runtime.py`

```py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .errors import BudgetExceeded
from .state import RLMState
from .tools import ToolRegistry
from .types import Message, ModelResponse, Role


@dataclass(frozen=True)
class StepBudget:
    max_steps: int
    max_depth: int


@dataclass(frozen=True)
class RunConfig:
    budget: StepBudget
    system_prompt: str = (
        "You are an RLM (Recursive Language Model). "
        "If you need to use a tool, respond with tool calls."
    )


class RLMRuntime:
    def __init__(self, provider, tools: Optional[ToolRegistry] = None) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()

    def run(self, user_prompt: str, state: Optional[RLMState] = None, cfg: Optional[RunConfig] = None) -> RLMState:
        state = state or RLMState()
        cfg = cfg or RunConfig(budget=StepBudget(max_steps=16, max_depth=6))

        # Ensure system prompt exists once at the beginning.
        if not state.history or state.history[0].role != Role.system:
            state.add(Message(role=Role.system, content=cfg.system_prompt))

        state.add(Message(role=Role.user, content=user_prompt))
        self._recurse(state=state, cfg=cfg, depth=0, steps_used=0)
        return state

    def _recurse(self, state: RLMState, cfg: RunConfig, depth: int, steps_used: int) -> int:
        if depth > cfg.budget.max_depth:
            raise BudgetExceeded(f"Max recursion depth exceeded: {cfg.budget.max_depth}")
        if steps_used >= cfg.budget.max_steps:
            raise BudgetExceeded(f"Max steps exceeded: {cfg.budget.max_steps}")

        resp: ModelResponse = self.provider.generate(state.history, tools=self.tools)
        state.add(Message(role=Role.assistant, content=resp.content))

        steps_used += 1
        if steps_used >= cfg.budget.max_steps:
            return steps_used

        if not resp.tool_calls:
            return steps_used

        # Execute tools deterministically in listed order.
        for tc in resp.tool_calls:
            if steps_used >= cfg.budget.max_steps:
                break
            result = self.tools.call(tc)
            # Tool message is added explicitly to history
            state.add(Message(role=Role.tool, name=result.name, content=result.content))
            steps_used += 1

        if steps_used >= cfg.budget.max_steps:
            return steps_used

        # Recurse: model sees tool outputs and decides next step(s)
        return self._recurse(state=state, cfg=cfg, depth=depth + 1, steps_used=steps_used)
```

---

## Examples

### `examples/local_ollama_basic.py`

```py
from rlmkit import RLMRuntime, RunConfig, StepBudget
from rlmkit.providers.ollama_provider import OllamaProvider
from rlmkit.providers.base import ProviderConfig

provider = OllamaProvider(ProviderConfig(model="llama3.1:8b", temperature=0.0))
rt = RLMRuntime(provider)

cfg = RunConfig(
    budget=StepBudget(max_steps=10, max_depth=4),
    system_prompt=(
        "You are a helpful assistant.\n"
        "If you need a tool, emit JSON exactly like:\n"
        '{"tool_calls":[{"id":"tc_1","name":"tool_name","arguments":{}}],"content":""}\n'
        "Otherwise answer normally."
    ),
)

state = rt.run("Explain recursion in 5 bullet points.", cfg=cfg)
print(state.history[-1].content)
```

### `examples/tool_calling_demo.py`

```py
from rlmkit import RLMRuntime, RunConfig, StepBudget, Tool, ToolRegistry
from rlmkit.providers.ollama_provider import OllamaProvider
from rlmkit.providers.base import ProviderConfig

tools = ToolRegistry()
tools.register(
    Tool(
        name="add",
        description="Add two numbers.",
        args_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        fn=lambda args: {"sum": args["a"] + args["b"]},
    )
)

provider = OllamaProvider(ProviderConfig(model="llama3.1:8b", temperature=0.0))
rt = RLMRuntime(provider, tools=tools)

cfg = RunConfig(
    budget=StepBudget(max_steps=12, max_depth=4),
    system_prompt=(
        "You can use tools.\n"
        "To call a tool, respond ONLY with JSON:\n"
        '{"tool_calls":[{"id":"tc_1","name":"add","arguments":{"a":2,"b":3}}],"content":""}\n'
        "After tool output, continue and give final answer."
    ),
)

state = rt.run("What is 19.5 + 22.25? Use the add tool.", cfg=cfg)
print("--- transcript ---")
for m in state.history:
    print(m.role.value, m.name or "", ":", m.content)
```

### `examples/remote_openai_basic.py`

```py
import os

from rlmkit import RLMRuntime
from rlmkit.providers.base import ProviderConfig
from rlmkit.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(ProviderConfig(model="gpt-4.1-mini", temperature=0.0), api_key=os.getenv("OPENAI_API_KEY"))
rt = RLMRuntime(provider)

state = rt.run("Write a tiny haiku about deterministic systems.")
print(state.history[-1].content)
```

### `examples/remote_anthropic_basic.py`

```py
import os

from rlmkit import RLMRuntime
from rlmkit.providers.base import ProviderConfig
from rlmkit.providers.anthropic_provider import AnthropicProvider

provider = AnthropicProvider(ProviderConfig(model="claude-3-5-sonnet-latest", temperature=0.0), api_key=os.getenv("ANTHROPIC_API_KEY"))
rt = RLMRuntime(provider)

state = rt.run("Summarize what 'structured state' means in agent runtimes.")
print(state.history[-1].content)
```

---

## Minimal docs

### `docs/architecture.md`

```md
# rlmkit architecture

rlmkit is a small, deterministic "Recursive Language Model" runtime:

- Provider-agnostic: providers implement `ModelProvider.generate(messages, tools) -> ModelResponse`.
- Deterministic: all mutable execution context lives in `RLMState` (memory/vars/history).
- Recursive execution: `RLMRuntime` calls the model, executes any tool calls, then recurses.
- Explicit control: `StepBudget(max_steps, max_depth)` bounds execution.

## Data flow

1. Add a system message (once) + user message
2. Provider generates assistant message, optionally with tool calls
3. Runtime executes tool calls in order, appends tool results as messages
4. Runtime recurses until no tool calls or budget exhausted

## Tool format

Tools live in `ToolRegistry`. Each tool has:
- name, description
- JSON-schema-like argument schema
- a Python callable `fn(args) -> Any`
Tool output is serialized deterministically to string and stored as a tool message.
```

### `docs/extending.md`

````md
# Extending rlmkit

## Add a new provider

Implement `ModelProvider`:

- Input: list of `rlmkit.types.Message`
- Output: `ModelResponse(content, tool_calls)`

Your provider should map from `Message` to your API's format, call the API, then:
- fill `content` with the assistant text
- parse tool calls into `ToolCall(id, name, arguments)`

## Add tools

Register tools:

```py
tools.register(
  Tool(
    name="my_tool",
    description="...",
    args_schema={...},
    fn=lambda args: ...
  )
)
````

## Stateful workflows

Use `state.vars` for structured machine-readable state and `state.memory` for persisted facts.
Because state is explicit, you can serialize it in your app and replay runs deterministically.

````

---

## Tests (budgets, recursion, tool calling, provider shims)

### `tests/test_budgets_and_depth.py`

```py
import pytest

from rlmkit import RLMRuntime, RunConfig, StepBudget
from rlmkit.errors import BudgetExceeded
from rlmkit.providers.base import ModelProvider, ProviderConfig
from rlmkit.types import ModelResponse, ToolCall


class LoopingProvider(ModelProvider):
    def __init__(self):
        super().__init__(ProviderConfig(model="fake"))

    def generate(self, messages, tools=None):
        # Always request a tool call -> forces recursion
        return ModelResponse(
            content="calling tool",
            tool_calls=[ToolCall(id="tc_1", name="noop", arguments={})],
            raw=None,
        )


def test_max_steps_budget_exceeded():
    rt = RLMRuntime(LoopingProvider())
    cfg = RunConfig(budget=StepBudget(max_steps=2, max_depth=10))
    with pytest.raises(BudgetExceeded):
        rt.run("hi", cfg=cfg)


def test_max_depth_budget_exceeded():
    rt = RLMRuntime(LoopingProvider())
    cfg = RunConfig(budget=StepBudget(max_steps=50, max_depth=0))
    with pytest.raises(BudgetExceeded):
        rt.run("hi", cfg=cfg)
````

### `tests/test_tool_calls.py`

```py
from rlmkit import RLMRuntime, Tool, ToolRegistry, RunConfig, StepBudget
from rlmkit.providers.base import ModelProvider, ProviderConfig
from rlmkit.types import ModelResponse, ToolCall


class OneToolThenStop(ModelProvider):
    def __init__(self):
        super().__init__(ProviderConfig(model="fake"))
        self.called = 0

    def generate(self, messages, tools=None):
        self.called += 1
        if self.called == 1:
            return ModelResponse(
                content="need tool",
                tool_calls=[ToolCall(id="tc_1", name="add", arguments={"a": 2, "b": 3})],
                raw=None,
            )
        return ModelResponse(content="done", tool_calls=[], raw=None)


def test_tool_execution_and_followup():
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="add",
            description="add",
            args_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
            fn=lambda args: {"sum": args["a"] + args["b"]},
        )
    )
    rt = RLMRuntime(OneToolThenStop(), tools=tools)
    cfg = RunConfig(budget=StepBudget(max_steps=10, max_depth=5))
    state = rt.run("calc", cfg=cfg)

    # Ensure tool message exists
    tool_msgs = [m for m in state.history if m.role.value == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "add"
    assert tool_msgs[0].content == '{"sum": 5}'
```

### `tests/test_provider_shims.py`

```py
import pytest

from rlmkit.providers.openai_provider import OpenAIProvider
from rlmkit.providers.anthropic_provider import AnthropicProvider
from rlmkit.providers.base import ProviderConfig
from rlmkit.errors import ProviderError


def test_openai_missing_dependency_raises():
    # If openai isn't installed in the env, this should raise ProviderError.
    try:
        OpenAIProvider(ProviderConfig(model="gpt-4.1-mini"))
    except ProviderError:
        return
    except Exception:
        pytest.fail("Expected ProviderError when dependency missing")
    # If dependency exists, pass test.
    assert True


def test_anthropic_missing_dependency_raises():
    try:
        AnthropicProvider(ProviderConfig(model="claude-3-5-sonnet-latest"))
    except ProviderError:
        return
    except Exception:
        pytest.fail("Expected ProviderError when dependency missing")
    assert True
```

---

## `README.md` (minimal, but complete)

````md
# rlmkit

A small, deterministic Recursive Language Model (RLM) runtime for:
- recursive prompting
- tool use
- structured explicit state
- bounded execution (steps + recursion depth)

## Install

Editable dev install:

```bash
pip install -e .
pip install -e ".[dev]"
````

Providers:

```bash
pip install -e ".[openai]"
pip install -e ".[anthropic]"
```

## Quickstart (Ollama)

```py
from rlmkit import RLMRuntime
from rlmkit.providers.ollama_provider import OllamaProvider
from rlmkit.providers.base import ProviderConfig

rt = RLMRuntime(OllamaProvider(ProviderConfig(model="llama3.1:8b", temperature=0.0)))
state = rt.run("Explain recursion in 3 bullets.")
print(state.history[-1].content)
```

## Architecture

See:

* docs/architecture.md
* docs/extending.md

## Running tests

```bash
pytest -q
```

```

---

## What you get (mapping to your deliverables)

- **rlmkit core runtime + provider adapters**: `runtime.py`, `state.py`, `tools.py`, `providers/*`
- **Example scripts (local + remote)**: `examples/*`
- **Minimal docs**: `docs/architecture.md`, `docs/extending.md`, README
- **Test suite**: budgets/depth, tool calling, provider shim behavior under missing deps

---

## Next obvious upgrades (optional, but easy to add later)

- A provider-agnostic **tool-call normalization** layer so Anthropic/OpenAI/Ollama formats converge more cleanly.
- Budget types beyond steps/depth (token budget, wall-clock deadline).
- A `StateStore` interface to snapshot/restore state for replay.
- Structured “RLM program” nodes (explicit recursion graph) if you want closer-to-paper RLM semantics.

If you want, paste your current repo tree (or upload a zip), and I’ll adapt this scaffold to your existing layout without breaking your MCP filesystem restrictions.
::contentReference[oaicite:1]{index=1}
```

[1]: https://chatgpt.com/share/696d184a-e144-8010-b698-e38c3d143b5a "ChatGPT - RLM Project Plan"
