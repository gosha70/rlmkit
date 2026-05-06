# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

r"""End-to-end test for ``wiki + rlm`` mode (issue gosha70/rlmkit#37 §A).

This is the differentiator test: it proves RLMKit's recursive
controller can drive the wiki ingest gate + draft loop. It uses a
``MockLLMClient`` to script the controller's responses so the test
is deterministic and runs without a real LLM call.

The flow exercised:

1. ``RLM.run(prompt=corpus, query=...)`` is invoked by
   :class:`RLMBackend.ingest`.
2. The controller hands the prompt to the mock client; the mock
   returns a single ``FINAL:`` step that contains the proposal
   JSON inside a fenced ``\`\`\`json`` block.
3. ``RLM`` extracts the FINAL answer and returns it as
   ``RLMResult.answer``.
4. ``RLMBackend._parse_proposal_json`` pulls the JSON out of the
   FINAL string and constructs an :class:`IngestProposal`.
5. The Ingestor runs the two-layer validation and writes the
   proposal file.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from rlmkit import RLM, RLMConfig, MockLLMClient
from rlmkit.wiki.backends import RLMBackend
from rlmkit.wiki.ingest import Ingestor
from rlmkit.wiki.proposal import IngestRequest
from rlmkit.wiki.schema import parse_frontmatter

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "knowledge" / "wiki" / "schema"


def _scripted_proposal_json() -> str:
    """A FINAL response containing a fenced JSON proposal block."""
    proposal = {
        "disposition": "accept",
        "reason": "incident with reusable lesson and citable source",
        "page_type": "incident",
        "slug": "rechunk-citation-loss",
        "title": "Rechunk Citation Loss",
        "draft_markdown": textwrap.dedent(
            """\
            ---
            page_type: incident
            slug: rechunk-citation-loss
            title: Rechunk Citation Loss
            status: draft
            last_reviewed: 2026-05-05
            sources:
              - issue: 99
            ---

            # Rechunk Citation Loss

            ## What happened

            A re-chunking changed retrieval scores enough that several
            citations vanished from the final answer.

            ## Why it happened

            Chunk boundaries shifted; affected chunks fell below the
            BM25 threshold without an embedding fallback to recover.

            ## What we changed

            Added a `chunk_size` invariant test and a sanity check
            that counts surviving citations after a re-chunk.

            ## How to recognize a recurrence

            Drop in citation count after a re-chunk on otherwise
            unchanged inputs.
            """
        ),
        "sources": [{"issue": 99}],
    }
    return "FINAL: ```json\n" + json.dumps(proposal, indent=2) + "\n```"


def test_rlm_backend_drives_recursive_controller_to_a_valid_proposal(tmp_path: Path) -> None:
    # Multi-source directory: the corpus the controller "explores".
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()
    (corpus_dir / "incident.md").write_text(
        "# Rechunk Citation Loss\n\nA re-chunking changed retrieval scores. Citations vanished.\n",
        encoding="utf-8",
    )
    (corpus_dir / "fix.md").write_text(
        "# Fix\n\nAdded chunk_size invariant + survival check.\n",
        encoding="utf-8",
    )

    # Scripted controller: one step that returns FINAL with the JSON.
    client = MockLLMClient([_scripted_proposal_json()])
    rlm = RLM(client=client, config=RLMConfig())

    # Run ingest end-to-end through the RLMBackend.
    out_dir = tmp_path / "proposals"
    ingestor = Ingestor(
        backend=RLMBackend(rlm=rlm),
        schema_dir=SCHEMA_DIR,
        output_dir=out_dir,
    )
    request = IngestRequest(source_path=corpus_dir, mode="rlm", backend_name="rlm")
    proposal_path = ingestor.run(request)

    # The proposal file exists and carries the right metadata.
    body = proposal_path.read_text(encoding="utf-8")
    fm, close = parse_frontmatter(body)
    assert fm["proposal_kind"] == "accept"
    assert fm["target_page_type"] == "incident"
    assert fm["target_slug"] == "rechunk-citation-loss"
    assert fm["backend"] == "rlm"

    # The candidate-wiki-page body (after the proposal frontmatter)
    # carries its own valid frontmatter.
    page_body = "\n".join(body.splitlines()[close:])
    page_fm, _ = parse_frontmatter(page_body)
    assert page_fm["page_type"] == "incident"
    assert page_fm["slug"] == "rechunk-citation-loss"

    # The mock client was actually invoked — proves the RLM
    # controller drove the loop, not just a passthrough. (The
    # corpus itself is loaded into the REPL as ``P`` rather than
    # the messages list, so we assert on call_count, not on the
    # message text.)
    assert client.call_count >= 1
    # The query (which embeds the schema excerpts) reached the
    # controller's user message.
    flat_messages = [m["content"] for call in client.call_history for m in call]
    assert any("four-question gate" in c for c in flat_messages)
