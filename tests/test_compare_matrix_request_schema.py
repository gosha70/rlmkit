"""AC-18 — the single `POST /api/chat/compare-matrix` endpoint accepts
the three new ranking_metric values under all three payload shapes.

Invalid values (e.g. `"bogus"`) still return 422 on every shape —
regression guard against accidentally widening the Literal to `str`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlmstudio.server.routes.compare_matrix import (
    CompareMatrixRequest,
    CompareMatrixRequestV2,
    CompareMatrixUnifiedRequest,
)

_NEW_METRICS = ("ttft", "decode_tokens_per_sec", "cache_hit_rate")


def _base_v1(metric: str) -> dict:
    return {
        "query": "q",
        "chat_provider_ids": ["cp-1"],
        "modes": ["direct"],
        "ranking_metric": metric,
    }


def _base_v2(metric: str) -> dict:
    return {
        "query": "q",
        "llm_provider_ids": ["llm-1"],
        "modes": ["direct"],
        "ranking_metric": metric,
    }


def _base_unified(metric: str) -> dict:
    # Supply chat_provider_ids so the model_validator passes.
    return {
        "query": "q",
        "chat_provider_ids": ["cp-1"],
        "modes": ["direct"],
        "ranking_metric": metric,
    }


@pytest.mark.parametrize("metric", _NEW_METRICS)
class TestNewMetricsAcceptedUnderAllShapes:
    def test_v1(self, metric: str):
        req = CompareMatrixRequest.model_validate(_base_v1(metric))
        assert req.ranking_metric == metric

    def test_v2(self, metric: str):
        req = CompareMatrixRequestV2.model_validate(_base_v2(metric))
        assert req.ranking_metric == metric

    def test_unified(self, metric: str):
        req = CompareMatrixUnifiedRequest.model_validate(_base_unified(metric))
        assert req.ranking_metric == metric


class TestBogusMetricRejected:
    """Regression guard — the Literal must not be widened to `str`."""

    def test_bogus_rejected_v1(self):
        with pytest.raises(ValidationError):
            CompareMatrixRequest.model_validate(_base_v1("bogus"))

    def test_bogus_rejected_v2(self):
        with pytest.raises(ValidationError):
            CompareMatrixRequestV2.model_validate(_base_v2("bogus"))

    def test_bogus_rejected_unified(self):
        with pytest.raises(ValidationError):
            CompareMatrixUnifiedRequest.model_validate(_base_unified("bogus"))
