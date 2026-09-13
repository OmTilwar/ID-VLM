"""
Unit tests for ID-VLM Ragas Multimodal Document Evaluation module.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragas_doc_eval import (
    IDVLMRagasEvaluator,
    MIDV_KYC_GOLD_BENCHMARK,
    clean_tokens
)

class TestIDVLMRagas:
    @pytest.fixture(scope="class")
    def evaluator(self):
        return IDVLMRagasEvaluator()

    def test_clean_tokens(self):
        tokens = clean_tokens("ESPAÑA - Documento Nacional 12345678Z")
        assert "documento" in tokens
        assert "12345678z" in tokens

    def test_context_precision_perfect(self, evaluator):
        retrieved = [(1, "Target Chunk", 0.95), (0, "Header Chunk", 0.40)]
        precision = evaluator.compute_context_precision(retrieved, target_idx=1)
        assert precision == 1.0

    def test_context_precision_second_rank(self, evaluator):
        retrieved = [(0, "Header Chunk", 0.95), (1, "Target Chunk", 0.40)]
        precision = evaluator.compute_context_precision(retrieved, target_idx=1)
        assert precision == 0.5

    def test_context_recall(self, evaluator):
        ctx = "Surname: TAMM, Personal Code: 38501150241, Citizenship: EST"
        key_facts = ["TAMM", "38501150241", "EST"]
        recall = evaluator.compute_context_recall(ctx, key_facts)
        assert recall == 1.0

    def test_faithfulness_grounded(self, evaluator):
        doc = "CISLO DOKLADU / DOC NO: EA849201 | PLATNOST: 10-06-2031"
        response = "Document Number: EA849201, Validity: 10-06-2031"
        faith = evaluator.compute_faithfulness(response, doc)
        assert faith == 1.0

    def test_faithfulness_hallucinated(self, evaluator):
        doc = "CISLO DOKLADU / DOC NO: EA849201 | PLATNOST: 10-06-2031"
        response = "Document Number: XX999999, Validity: 10-06-2099"  # Hallucinated tokens
        faith = evaluator.compute_faithfulness(response, doc)
        assert faith == 0.0

    def test_full_evaluation_run(self, evaluator):
        res = evaluator.evaluate_model("finetuned")
        assert "context_precision" in res
        assert "context_recall" in res
        assert "faithfulness" in res
        assert "answer_relevancy" in res
        assert res["faithfulness"] > 0.60
