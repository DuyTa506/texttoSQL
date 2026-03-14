"""
Tests for CorrectionFormatter – verifies serialization, GRPO format conversion,
and CorrectionDataset load/save/filter operations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from training.correction_formatter import (
    CorrectionDataset,
    CorrectionSample,
    _ERROR_HINTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sample(**kwargs) -> CorrectionSample:
    defaults = dict(
        question="What are all student names?",
        schema_context="CREATE TABLE students (id INTEGER, name TEXT)",
        wrong_sql="SELECT * FORM students",
        error_type="syntax_error",
        error_message="near 'FORM': syntax error",
        corrected_sql="SELECT name FROM students",
        correction_reasoning="The keyword FROM was misspelled as FORM.",
        db_id="test_db",
        difficulty="medium",
    )
    defaults.update(kwargs)
    return CorrectionSample(**defaults)


# ---------------------------------------------------------------------------
# CorrectionSample tests
# ---------------------------------------------------------------------------


class TestCorrectionSample:
    def test_to_dict_roundtrip(self):
        sample = make_sample()
        d = sample.to_dict()
        restored = CorrectionSample.from_dict(d)
        assert restored.question == sample.question
        assert restored.wrong_sql == sample.wrong_sql
        assert restored.corrected_sql == sample.corrected_sql
        assert restored.error_type == sample.error_type
        assert restored.db_id == sample.db_id

    def test_from_dict_handles_missing_fields(self):
        d = {
            "question": "q",
            "wrong_sql": "SELECT * FORM t",
            "corrected_sql": "SELECT * FROM t",
        }
        sample = CorrectionSample.from_dict(d)
        assert sample.schema_context == ""
        assert sample.error_type == "execution_error"
        assert sample.correction_reasoning == ""
        assert sample.db_id == ""
        assert sample.difficulty == "unknown"

    def test_to_grpo_dict_structure(self):
        sample = make_sample()
        grpo = sample.to_grpo_dict()

        assert "prompt" in grpo
        assert "answer" in grpo
        assert "task_type" in grpo
        assert "wrong_sql" in grpo
        assert "error_type" in grpo

        assert grpo["task_type"] == "correction"
        assert grpo["answer"] == sample.corrected_sql
        assert grpo["wrong_sql"] == sample.wrong_sql
        assert grpo["error_type"] == sample.error_type

    def test_to_grpo_dict_prompt_contains_key_fields(self):
        sample = make_sample()
        grpo = sample.to_grpo_dict()
        prompt = grpo["prompt"]

        # Prompt must contain the schema, question, wrong SQL, and error info
        assert sample.schema_context in prompt
        assert sample.question in prompt
        assert sample.wrong_sql in prompt
        assert sample.error_type in prompt

    def test_to_grpo_dict_includes_hint(self):
        sample = make_sample(error_type="syntax_error")
        grpo = sample.to_grpo_dict()
        prompt = grpo["prompt"]
        # Should include the hint for syntax_error
        assert "syntax" in prompt.lower()

    def test_all_error_types_have_hints(self):
        error_types = [
            "syntax_error", "no_such_table", "no_such_column",
            "wrong_result", "empty_result", "execution_error"
        ]
        for etype in error_types:
            assert etype in _ERROR_HINTS, f"Missing hint for {etype}"
            assert len(_ERROR_HINTS[etype]) > 10, f"Hint too short for {etype}"

    def test_unknown_error_type_uses_default_hint(self):
        sample = make_sample(error_type="unknown_error_xyz")
        grpo = sample.to_grpo_dict()
        prompt = grpo["prompt"]
        # Should not raise; uses fallback hint
        assert "sql" in prompt.lower() or "review" in prompt.lower()


# ---------------------------------------------------------------------------
# CorrectionDataset tests
# ---------------------------------------------------------------------------


class TestCorrectionDataset:
    def test_save_and_load(self, tmp_path):
        samples = [
            make_sample(wrong_sql=f"SELECT FORM t WHERE id = {i}", db_id=f"db_{i}")
            for i in range(5)
        ]
        dataset = CorrectionDataset(samples)
        path = tmp_path / "correction.jsonl"
        dataset.save(path)

        loaded = CorrectionDataset.load(path)
        assert len(loaded) == 5
        assert loaded.samples[0].db_id == "db_0"
        assert loaded.samples[4].db_id == "db_4"

    def test_load_skips_empty_lines(self, tmp_path):
        path = tmp_path / "correction.jsonl"
        sample = make_sample()
        with open(path, "w") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")
            f.write("\n")  # empty line
            f.write("\n")  # another empty line
            f.write(json.dumps(sample.to_dict()) + "\n")

        loaded = CorrectionDataset.load(path)
        assert len(loaded) == 2

    def test_to_grpo_list(self):
        samples = [make_sample() for _ in range(3)]
        dataset = CorrectionDataset(samples)
        grpo_list = dataset.to_grpo_list()

        assert len(grpo_list) == 3
        for item in grpo_list:
            assert item["task_type"] == "correction"
            assert "prompt" in item
            assert "answer" in item

    def test_filter_by_error_type(self):
        samples = [
            make_sample(error_type="syntax_error"),
            make_sample(error_type="wrong_result"),
            make_sample(error_type="syntax_error"),
            make_sample(error_type="no_such_table"),
        ]
        dataset = CorrectionDataset(samples)

        syntax_only = dataset.filter_by_error_type(["syntax_error"])
        assert len(syntax_only) == 2

        mixed = dataset.filter_by_error_type(["syntax_error", "wrong_result"])
        assert len(mixed) == 3

        empty = dataset.filter_by_error_type(["nonexistent"])
        assert len(empty) == 0

    def test_filter_by_difficulty(self):
        samples = [
            make_sample(difficulty="easy"),
            make_sample(difficulty="medium"),
            make_sample(difficulty="hard"),
            make_sample(difficulty="medium"),
        ]
        dataset = CorrectionDataset(samples)

        medium_hard = dataset.filter_by_difficulty(["medium", "hard"])
        assert len(medium_hard) == 3

    def test_len(self):
        samples = [make_sample() for _ in range(7)]
        dataset = CorrectionDataset(samples)
        assert len(dataset) == 7

    def test_repr_shows_counts(self):
        samples = [
            make_sample(error_type="syntax_error"),
            make_sample(error_type="wrong_result"),
            make_sample(error_type="syntax_error"),
        ]
        dataset = CorrectionDataset(samples)
        r = repr(dataset)
        assert "CorrectionDataset" in r
        assert "syntax_error" in r

    def test_save_creates_parent_dirs(self, tmp_path):
        samples = [make_sample()]
        dataset = CorrectionDataset(samples)
        nested_path = tmp_path / "a" / "b" / "c" / "out.jsonl"
        dataset.save(nested_path)
        assert nested_path.exists()
