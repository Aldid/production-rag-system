"""Integration test executing the full 100-case evaluation benchmark harness."""

import os
import pytest
from src.evals.harness import EvaluationHarness


def test_full_100_case_benchmark(tmp_path):
    harness = EvaluationHarness()
    chunks_count = harness.setup_corpus()
    assert chunks_count >= 5

    summary, results = harness.run_benchmark()

    # Core evaluation assertions
    assert summary.total_cases == 100
    assert len(results) == 100

    # Information Retrieval metrics
    assert summary.hit_rate_at_1 >= 0.80, f"HitRate@1 below 80%: {summary.hit_rate_at_1}"
    assert summary.hit_rate_at_3 >= 0.95, f"HitRate@3 below 95%: {summary.hit_rate_at_3}"
    assert summary.hit_rate_at_5 >= 0.98, f"HitRate@5 below 98%: {summary.hit_rate_at_5}"
    assert summary.mean_reciprocal_rank >= 0.85, f"MRR below 0.85: {summary.mean_reciprocal_rank}"

    # Answer Quality & Grounding metrics
    assert summary.avg_keyword_coverage >= 0.80, f"Keyword coverage below 80%: {summary.avg_keyword_coverage}"
    assert summary.avg_faithfulness >= 0.85, f"Faithfulness below 85%: {summary.avg_faithfulness}"
    assert summary.avg_citation_precision >= 0.90, f"Citation precision below 90%: {summary.avg_citation_precision}"

    # Latency SLAs
    assert summary.avg_latency_ms < 50.0, f"Average latency exceeded 50ms: {summary.avg_latency_ms}"

    # Export report to a temporary directory (keeps the test hermetic and OS-independent)
    report_path = harness.export_report(summary, results, output_dir=str(tmp_path))
    assert os.path.exists(report_path)
    assert os.path.exists(tmp_path / "evaluation_report.json")
