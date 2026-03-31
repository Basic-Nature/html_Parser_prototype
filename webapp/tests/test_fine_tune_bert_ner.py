"""Tests for BERT NER training utilities."""

from __future__ import annotations

import sys
import types


if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForTokenClassification = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.DataCollatorForTokenClassification = object
    transformers_stub.Trainer = object
    transformers_stub.TrainingArguments = object
    sys.modules["transformers"] = transformers_stub


from webapp.parser.health.fine_tune_bert_ner import align_entity_spans_to_tokens


class TestAlignEntitySpansToTokens:
    def test_aligns_single_token_entity(self):
        tokens, ner_tags = align_entity_spans_to_tokens(
            "Alice won decisively",
            [{"start": 0, "end": 5, "label": "PERSON"}],
        )

        assert tokens == ["Alice", "won", "decisively"]
        assert ner_tags == ["B-PERSON", "O", "O"]

    def test_aligns_multi_token_entity_with_bio_tags(self):
        text = "New York County reported results"
        tokens, ner_tags = align_entity_spans_to_tokens(
            text,
            [{"start": 0, "end": 15, "label": "GPE"}],
        )

        assert tokens == ["New", "York", "County", "reported", "results"]
        assert ner_tags == ["B-GPE", "I-GPE", "I-GPE", "O", "O"]

    def test_ignores_invalid_or_empty_entity_spans(self):
        tokens, ner_tags = align_entity_spans_to_tokens(
            "Contest results pending",
            [
                {"start": 0, "end": 0, "label": "CONTEST"},
                {"start": 2, "end": 1, "label": "CONTEST"},
                {"start": 0, "end": 7, "label": ""},
            ],
        )

        assert tokens == ["Contest", "results", "pending"]
        assert ner_tags == ["O", "O", "O"]

    def test_only_tags_tokens_overlapping_entity_span(self):
        text = "Alice Johnson for Mayor"
        tokens, ner_tags = align_entity_spans_to_tokens(
            text,
            [{"start": 0, "end": 13, "label": "PERSON"}],
        )

        assert tokens == ["Alice", "Johnson", "for", "Mayor"]
        assert ner_tags == ["B-PERSON", "I-PERSON", "O", "O"]
