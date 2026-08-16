"""Bundled RLM Studio UI directory.

This package exists as a placeholder so that `pip install rlm-studio` always
creates the ``rlmstudio/_ui/`` directory, regardless of whether the
frontend static export has been built. The build pipeline (see
``doc_internal/release/steps/04-publish-and-distribute.md``) populates
this directory with the Next.js static export before ``uv build`` runs.

If you are reading this file as the only contents of ``rlmstudio/_ui/``,
the frontend has not been built. See `RELEASING.md` (or its internal
equivalent) for the build procedure.
"""
