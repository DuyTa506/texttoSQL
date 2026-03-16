"""
Tests for ValueScanner – verifies value matching, similarity scoring, and hint formatting.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.schema.models import Column, Database, Table
from src.retrieval.value_scanner import ValueMatch, ValueScanner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_sqlite_db(tmp_path):
    """Create a small in-memory-style SQLite DB for testing."""
    db_path = tmp_path / "test_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, city TEXT)"
    )
    cursor.execute(
        "INSERT INTO students VALUES "
        "(1, 'Alice Johnson', 'New York'), "
        "(2, 'Bob Smith', 'Los Angeles'), "
        "(3, 'Carol White', 'Chicago')"
    )
    cursor.execute(
        "CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, department TEXT)"
    )
    cursor.execute(
        "INSERT INTO courses VALUES "
        "(1, 'Linear Algebra', 'Mathematics'), "
        "(2, 'Python Programming', 'Computer Science')"
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def test_database(temp_sqlite_db):
    """Build a Database object pointing to the temp SQLite file."""
    students = Table(
        name="students",
        columns=[
            Column(name="id", dtype="INTEGER", primary_key=True),
            Column(name="name", dtype="TEXT"),
            Column(name="city", dtype="TEXT"),
        ],
    )
    courses = Table(
        name="courses",
        columns=[
            Column(name="id", dtype="INTEGER", primary_key=True),
            Column(name="title", dtype="TEXT"),
            Column(name="department", dtype="TEXT"),
        ],
    )
    return Database(
        db_id="test_db",
        tables=[students, courses],
        db_path=str(temp_sqlite_db),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValueScannerBasic:
    def test_exact_match(self, test_database):
        scanner = ValueScanner(max_values_per_col=100, top_k=5, min_score=0.75)
        matches = scanner.scan("Who lives in Chicago?", test_database)
        assert len(matches) >= 1
        matched_values = {m.matched_value.lower() for m in matches}
        assert "chicago" in matched_values

    def test_exact_match_for_name(self, test_database):
        scanner = ValueScanner(max_values_per_col=100, top_k=5, min_score=0.9)
        matches = scanner.scan("Find students named Alice Johnson", test_database)
        assert len(matches) >= 1

    def test_returns_empty_without_db_path(self):
        scanner = ValueScanner()
        db = Database(db_id="empty", db_path=None)
        matches = scanner.scan("any question", db)
        assert matches == []

    def test_returns_empty_for_no_match(self, test_database):
        scanner = ValueScanner(max_values_per_col=100, top_k=5, min_score=0.99)
        matches = scanner.scan("zzzzzzz xxxxxxxxxxx", test_database)
        assert matches == []

    def test_top_k_respected(self, test_database):
        scanner = ValueScanner(max_values_per_col=100, top_k=2, min_score=0.5)
        matches = scanner.scan(
            "New York Los Angeles Chicago Alice Bob Carol Mathematics", test_database
        )
        assert len(matches) <= 2

    def test_skips_numeric_columns(self, test_database):
        scanner = ValueScanner(max_values_per_col=100, top_k=10, min_score=0.9)
        # Integer column values like "1", "2", "3" should not be returned
        matches = scanner.scan("Show me student 1", test_database)
        # Matches should be from TEXT columns, not INTEGER id
        for m in matches:
            assert m.column_name != "id"


class TestValueScannerSimilarity:
    def test_exact_score_is_one(self):
        score = ValueScanner._similarity("Alice", "Alice")
        assert score == 1.0

    def test_case_insensitive(self):
        score = ValueScanner._similarity("alice", "Alice")
        assert score == 1.0

    def test_substring_match(self):
        score = ValueScanner._similarity("New York", "New York City")
        assert score > 0.5

    def test_low_score_for_different_strings(self):
        score = ValueScanner._similarity("xyz123", "abcdef")
        assert score < 0.5

    def test_empty_string(self):
        score = ValueScanner._similarity("", "hello")
        assert score == 0.0


class TestValueScannerCandidateExtraction:
    def test_extracts_quoted_strings(self):
        candidates = ValueScanner._extract_candidates('Find "Alice Johnson" in db')
        assert "Alice Johnson" in candidates

    def test_extracts_capitalized_phrases(self):
        candidates = ValueScanner._extract_candidates(
            "Students from New York who studied Computer Science"
        )
        assert any("New York" in c for c in candidates)

    def test_extracts_individual_tokens(self):
        candidates = ValueScanner._extract_candidates("Find students from Chicago")
        assert "Chicago" in candidates or "chicago" in [c.lower() for c in candidates]

    def test_deduplicates(self):
        candidates = ValueScanner._extract_candidates("Alice and alice and ALICE")
        lower_candidates = [c.lower() for c in candidates]
        assert lower_candidates.count("alice") == 1


class TestToSchemaHints:
    def test_formats_correctly(self):
        scanner = ValueScanner()
        matches = [
            ValueMatch("students", "city", "New York", 1.0),
            ValueMatch("courses", "department", "Mathematics", 0.9),
        ]
        hints = scanner.to_schema_hints(matches)
        assert "-- students.city likely contains: 'New York'" in hints
        assert "-- courses.department likely contains: 'Mathematics'" in hints

    def test_empty_matches(self):
        scanner = ValueScanner()
        hints = scanner.to_schema_hints([])
        assert hints == ""


class TestValueMatchDataclass:
    def test_repr(self):
        m = ValueMatch("t", "c", "val", 0.95)
        assert "t.c" in repr(m)
        assert "val" in repr(m)
