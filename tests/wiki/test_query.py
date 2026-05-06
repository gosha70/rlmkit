# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki-first query path tests (Mode B + Mode D)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from rlmkit import RLM, RLMConfig, MockLLMClient
from rlmkit.wiki.query import INSUFFICIENT, query_wiki

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed_wiki(tmp_path: Path) -> Path:
    src_wiki = REPO_ROOT / "knowledge" / "wiki"
    dst = tmp_path / "wiki"
    shutil.copytree(src_wiki, dst)
    return dst


def test_query_wiki_returns_top_page_when_no_rlm_provided(tmp_path: Path) -> None:
    wiki = _seed_wiki(tmp_path)
    result = query_wiki("what is an LLM Wiki knowledge layer", wiki)
    assert "knowledge-layer" in result.pages_consulted[0]
    assert "LLM Wiki" in result.answer


def test_query_wiki_returns_insufficient_on_total_miss(tmp_path: Path) -> None:
    wiki = _seed_wiki(tmp_path)
    result = query_wiki("zzzzzzz totally unrelated keywords here", wiki)
    assert result.answer == INSUFFICIENT
    assert result.pages_consulted == []


def test_query_wiki_falls_back_to_rlm_when_wiki_signals_insufficient(tmp_path: Path) -> None:
    wiki = _seed_wiki(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text("# Raw\n\nThe specific raw answer is 42.\n", encoding="utf-8")

    # First mock response: the wiki-first LLM call signals INSUFFICIENT.
    # Second: the RLM controller's FINAL answer (one step).
    client = MockLLMClient([INSUFFICIENT, "FINAL: 42"])
    rlm = RLM(client=client, config=RLMConfig())

    result = query_wiki(
        "what is the LLM Wiki specific raw answer", wiki, raw_dir=raw, rlm=rlm
    )
    assert result.fell_back_to_rlm
    assert "42" in result.answer
    assert result.rlm_trace is not None
    assert len(result.rlm_trace) >= 1


def test_query_wiki_does_not_escalate_when_wiki_answer_is_sufficient(tmp_path: Path) -> None:
    wiki = _seed_wiki(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()

    client = MockLLMClient(["The wiki already has the answer."])
    rlm = SimpleNamespace(client=client, run=lambda prompt, query: None)

    result = query_wiki(
        "what is an LLM Wiki knowledge layer", wiki, raw_dir=raw, rlm=rlm
    )
    assert not result.fell_back_to_rlm
    assert "wiki already has" in result.answer
