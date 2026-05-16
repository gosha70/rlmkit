# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""End-to-end tests for the LLM Wiki backbone.

Round-trips ingest → promote → query against a tmp-fixture wiki
using the deterministic in-process backend so the suite needs no
network access and no model. The wiki + rlm fallback path is
exercised with a fake LLMClient that returns a final-answer block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmkit.application.sandbox_vars import MODE_WIKI, MODE_WIKI_RLM
from rlmkit.config import RLMConfig
from rlmkit.strategies import (
    LLMStrategy,
    StrategyResult,
    WikiRLMStrategy,
    WikiStrategy,
)
from rlmkit.strategies.wiki.backends import (
    DeterministicTestBackend,
    LLMClientWikiBackend,
)
from rlmkit.strategies.wiki.health_lint import lint_health
from rlmkit.strategies.wiki.ingestor import ingest
from rlmkit.strategies.wiki.promoter import promote
from rlmkit.strategies.wiki.querier import query
from rlmkit.strategies.wiki.structural_lint import lint


def _seed_wiki(wiki_dir: Path) -> None:
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "index.md").write_text(
        "---\n"
        "page_type: index\n"
        "slug: index\n"
        "title: Wiki Index\n"
        "status: stable\n"
        "last_reviewed: 2026-05-07\n"
        "---\n\n"
        "# Wiki Index\n\n"
        "## Pages\n\n"
        "- [overview](overview.md)\n"
        "- [karpathy wiki](concepts/karpathy-wiki.md)\n",
        encoding="utf-8",
    )
    (wiki_dir / "log.md").write_text(
        "---\n"
        "page_type: log\n"
        "slug: log\n"
        "title: Log\n"
        "status: stable\n"
        "last_reviewed: 2026-05-07\n"
        "---\n\n"
        "# Wiki log\n",
        encoding="utf-8",
    )
    (wiki_dir / "overview.md").write_text(
        "---\n"
        "page_type: overview\n"
        "slug: overview\n"
        "title: Overview\n"
        "status: stable\n"
        "last_reviewed: 2026-05-07\n"
        "sources:\n"
        "  - path: README.md\n"
        "---\n\n"
        "## Summary\n\n"
        "Test fixture wiki for the rlmkit wiki strategy suite.\n\n"
        "## Key ideas\n\n"
        "- Round-trips ingest, promote, query.\n\n"
        "## Where this shows up\n\n"
        "- tests/test_wiki_strategy.py\n\n"
        "## Related\n\n"
        "- [index](index.md)\n",
        encoding="utf-8",
    )
    # Seed a concept page that is reachable ONLY from index.md so the
    # weak-orphan check has something to flag.
    (wiki_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (wiki_dir / "concepts" / "karpathy-wiki.md").write_text(
        "---\n"
        "page_type: concept\n"
        "slug: karpathy-wiki\n"
        "title: Karpathy LLM Wiki\n"
        "status: stable\n"
        "last_reviewed: 2026-05-07\n"
        "sources:\n"
        "  - url: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f\n"
        "    retrieved: 2026-05-07\n"
        "---\n\n"
        "## Summary\n\n"
        "The persistent, distilled markdown layer pattern.\n\n"
        "## Key ideas\n\n"
        "- Index-first navigation.\n\n"
        "## Where this shows up\n\n"
        "- src/rlmkit/strategies/wiki/\n\n"
        "## Related\n\n"
        "- [index](../index.md)\n",
        encoding="utf-8",
    )


@pytest.fixture
def fixture_wiki(tmp_path: Path) -> Path:
    wiki_dir = tmp_path / "knowledge" / "wiki"
    _seed_wiki(wiki_dir)
    return wiki_dir


def test_strategy_protocol_compliance() -> None:
    """WikiStrategy and WikiRLMStrategy both satisfy LLMStrategy structurally."""
    # Structural compliance check (Protocol isinstance is unreliable).
    assert hasattr(WikiStrategy, "name")
    assert callable(getattr(WikiStrategy, "run", None))
    assert hasattr(WikiRLMStrategy, "name")
    assert callable(getattr(WikiRLMStrategy, "run", None))


def test_wiki_strategy_runs_query_against_fixture(fixture_wiki: Path) -> None:
    backend = DeterministicTestBackend()
    strat = WikiStrategy(wiki_dir=fixture_wiki, backend=backend)
    assert strat.name == MODE_WIKI

    result: StrategyResult = strat.run(
        content="ignored",
        query="What does the wiki say about overview?",
    )
    assert result.success
    assert result.strategy == MODE_WIKI
    assert "consulted" in result.answer.lower()
    assert result.metadata["citation_count"] >= 1
    assert "overview.md" in result.metadata["pages_loaded"]


def test_ingest_promote_query_round_trip(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "knowledge" / "wiki"
    proposals = tmp_path / "doc_internal" / "proposals"
    _seed_wiki(wiki_dir)

    source = tmp_path / "src.md"
    source.write_text(
        "# Origin Alignment\n\nThe user's origin governs the build. This source explains why.\n",
        encoding="utf-8",
    )

    backend = DeterministicTestBackend()
    patch, proposal_dir = ingest(
        source_path=source,
        wiki_dir=wiki_dir,
        proposals_root=proposals,
        backend=backend,
    )
    assert proposal_dir.exists()
    assert (proposal_dir / "plan.json").exists()
    create_paths = [e.path for e in patch.edits if e.action == "create"]
    assert any(p.startswith("concepts/") for p in create_paths)

    result = promote(
        proposal_dir=proposal_dir,
        wiki_dir=wiki_dir,
        archive_root=proposals / ".applied",
        dry_run=False,
    )
    assert not result.dry_run
    assert any(p.startswith("concepts/") for p in result.applied_paths)
    # Live wiki has the new page now.
    assert any(wiki_dir.glob("concepts/*.md"))

    # Index updated, log appended.
    index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")
    log_text = (wiki_dir / "log.md").read_text(encoding="utf-8")
    assert "concepts/" in index_text
    # Log appended at least one new line beyond the seed body.
    assert log_text.count("\n") > 5

    # Query reads the wiki and finds the new page.
    audit = tmp_path / "wiki-query-log.jsonl"
    ans = query(
        question="origin alignment",
        wiki_dir=wiki_dir,
        backend=backend,
        audit_log_path=audit,
    )
    assert ans.pages_loaded
    assert audit.exists()
    log_line = json.loads(audit.read_text(encoding="utf-8").splitlines()[0])
    assert log_line["question"] == "origin alignment"
    assert log_line["pages_loaded"]


def test_promote_rejects_invalid_patch(tmp_path: Path) -> None:
    """If the structural lint fails, the wiki is not modified."""
    wiki_dir = tmp_path / "knowledge" / "wiki"
    proposals = tmp_path / "proposals"
    _seed_wiki(wiki_dir)

    proposal_dir = proposals / "2026-05-07-bad"
    (proposal_dir / "preview" / "concepts").mkdir(parents=True)
    bad_page = proposal_dir / "preview" / "concepts" / "bogus.md"
    # Frontmatter is missing required keys.
    bad_page.write_text("---\npage_type: concept\n---\n\nbody\n", encoding="utf-8")
    (proposal_dir / "plan.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_path": "x.md",
                "rationale": "bad fixture",
                "edits": [
                    {
                        "path": "concepts/bogus.md",
                        "action": "create",
                        "preview": "preview/concepts/bogus.md",
                        "rationale": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from rlmkit.strategies.wiki.errors import PromoteValidationError

    with pytest.raises(PromoteValidationError):
        promote(
            proposal_dir=proposal_dir,
            wiki_dir=wiki_dir,
            archive_root=proposals / ".applied",
        )
    # Live wiki untouched.
    assert not (wiki_dir / "concepts" / "bogus.md").exists()


def test_structural_lint_clean_on_seed(fixture_wiki: Path) -> None:
    violations = lint(fixture_wiki)
    assert violations == [], f"unexpected violations: {violations}"


def test_health_lint_runs_without_backend(fixture_wiki: Path) -> None:
    findings = lint_health(fixture_wiki, repo_root=fixture_wiki.parent.parent)
    # Seed wiki has overview reachable from index (1 inbound), expected
    # to be a weak-orphan finding.
    kinds = {f.kind for f in findings}
    assert "weak-orphan" in kinds


# ---------------------------------------------------------------------------
# WikiRLMStrategy fallback path
# ---------------------------------------------------------------------------


class _FakeFinalAnswerClient:
    """LLMClient fake that emits a single FINAL_ANSWER step.

    Matches the contract of ``rlmkit.core.parsing.parse_response`` so
    the RLM controller terminates after one step.
    """

    def __init__(self, answer: str = "fallback synthesis OK") -> None:
        self.answer = answer
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return f"FINAL: {self.answer}"


def test_wiki_rlm_strategy_falls_back_when_forced(fixture_wiki: Path) -> None:
    backend = DeterministicTestBackend()
    client = _FakeFinalAnswerClient(answer="recursive synth from wiki + rlm")
    strat = WikiRLMStrategy(
        wiki_dir=fixture_wiki,
        client=client,
        backend=backend,
        rlm_config=RLMConfig(),
        force_rlm=True,
    )
    assert strat.name == MODE_WIKI_RLM
    result = strat.run(content="", query="what is the overview?")
    assert result.metadata["fallback"] == "rlm"
    assert client.calls >= 1
    # The RLM-produced answer should be the strategy's answer, not
    # the wiki strategy's stub.
    assert "recursive synth from wiki + rlm" in result.answer


def test_wiki_rlm_strategy_skips_fallback_when_coverage_strong(
    fixture_wiki: Path,
) -> None:
    backend = DeterministicTestBackend()
    client = _FakeFinalAnswerClient(answer="should-not-be-called")
    strat = WikiRLMStrategy(
        wiki_dir=fixture_wiki,
        client=client,
        backend=backend,
        rlm_config=RLMConfig(),
        force_rlm=False,
        # Test backend always returns at least 1 citation when pages
        # are loaded; coverage is "strong" by default.
        min_citations=1,
    )
    result = strat.run(content="", query="what is the overview?")
    assert result.metadata["fallback"] == "none"
    # RLM client should not have been called.
    assert client.calls == 0


# ---------------------------------------------------------------------------
# CLI smoke (covers argparse + dispatch).
# ---------------------------------------------------------------------------


def test_cli_lint_reports_clean(fixture_wiki: Path, tmp_path: Path, capsys, monkeypatch) -> None:
    from rlmkit.cli.wiki import main

    monkeypatch.chdir(fixture_wiki.parent.parent)
    rc = main(["lint", "--wiki-root", str(fixture_wiki)])
    out = capsys.readouterr().out
    assert "structural: 0 violations" in out
    assert rc == 0


def test_cli_ingest_writes_proposal(tmp_path: Path, capsys, monkeypatch) -> None:
    from rlmkit.cli.wiki import main

    wiki_dir = tmp_path / "knowledge" / "wiki"
    proposals = tmp_path / "doc_internal" / "proposals"
    _seed_wiki(wiki_dir)
    source = tmp_path / "src.md"
    source.write_text("# Karpathy\nWiki pattern.\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "ingest",
            str(source),
            "--wiki-root",
            str(wiki_dir),
            "--proposals-root",
            str(proposals),
            "--backend",
            "test",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "proposal:" in out
    assert "[create]" in out
