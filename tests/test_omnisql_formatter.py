"""Tests for OmniSQLFormatter — parsing and conversion of OmniSQL data."""

import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_formatter import OmniSQLFormatter, TrainingSample


# ============================================================================
# Fixtures
# ============================================================================

SAMPLE_INPUT_SEQ = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
CREATE TABLE head (
    head_ID number, -- example: [1, 2]
    age number, -- example: [67.0, 68.0]
    PRIMARY KEY (head_ID)
);

CREATE TABLE department (
    Department_ID number, -- example: [1, 2]
    Name text, -- example: ['State', 'Treasury']
    PRIMARY KEY (Department_ID)
);
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
How many heads of the departments are older than 56 ?

Instructions:
- Make sure you only output the information that is asked in the question.
- The generated query should return all of the information asked in the question.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

SAMPLE_OUTPUT_SEQ = """To translate the natural language question into an executable SQLite query, let's break it down step by step:

---

### Step 1: Understand the Question
The question asks us to determine how many "heads of the departments" are older than 56.

### Step 2: Identify Relevant Tables and Columns
From the database schema:
- The `head` table contains the information about department heads, including their `age`.

### Step 3: Write the SQL Query
The SQL query to answer the question is:

```sql
SELECT COUNT(*) 
FROM head 
WHERE age > 56;
```

### Step 4: Verify the Query
This query is efficient and directly addresses the natural language question."""


# ============================================================================
# Tests: Input Parsing
# ============================================================================

class TestParseInputSeq:
    def test_extract_schema(self):
        schema, question = OmniSQLFormatter._parse_input_seq(SAMPLE_INPUT_SEQ)
        assert "CREATE TABLE head" in schema
        assert "CREATE TABLE department" in schema

    def test_extract_question(self):
        schema, question = OmniSQLFormatter._parse_input_seq(SAMPLE_INPUT_SEQ)
        assert "How many heads of the departments are older than 56" in question

    def test_schema_excludes_instructions(self):
        schema, question = OmniSQLFormatter._parse_input_seq(SAMPLE_INPUT_SEQ)
        assert "Instructions:" not in schema
        assert "Task Overview:" not in schema

    def test_question_excludes_instructions(self):
        schema, question = OmniSQLFormatter._parse_input_seq(SAMPLE_INPUT_SEQ)
        assert "Make sure you only output" not in question

    def test_empty_input(self):
        schema, question = OmniSQLFormatter._parse_input_seq("")
        assert schema == ""
        assert question == ""

    def test_partial_input_no_question(self):
        partial = "Task Overview:\n...\nDatabase Schema:\nCREATE TABLE x (id INT);\nThis schema describes..."
        schema, question = OmniSQLFormatter._parse_input_seq(partial)
        assert "CREATE TABLE x" in schema


# ============================================================================
# Tests: Output Parsing
# ============================================================================

class TestParseOutputSeq:
    def test_extract_sql(self):
        reasoning, sql = OmniSQLFormatter._parse_output_seq(SAMPLE_OUTPUT_SEQ)
        assert "SELECT COUNT(*)" in sql
        assert "FROM head" in sql
        assert "WHERE age > 56" in sql

    def test_extract_reasoning(self):
        reasoning, sql = OmniSQLFormatter._parse_output_seq(SAMPLE_OUTPUT_SEQ)
        assert "Step 1" in reasoning
        assert "Understand the Question" in reasoning

    def test_sql_not_in_reasoning(self):
        reasoning, sql = OmniSQLFormatter._parse_output_seq(SAMPLE_OUTPUT_SEQ)
        # The reasoning should not contain the ```sql``` block itself
        assert "```sql" not in reasoning

    def test_no_code_block_fallback(self):
        output = "The answer is: SELECT * FROM test WHERE id = 1;"
        reasoning, sql = OmniSQLFormatter._parse_output_seq(output)
        assert "SELECT" in sql

    def test_multiple_code_blocks_takes_last(self):
        output = """Some reasoning
```sql
SELECT 1;
```
More reasoning
```sql
SELECT COUNT(*) FROM final_table;
```
"""
        reasoning, sql = OmniSQLFormatter._parse_output_seq(output)
        assert "final_table" in sql

    def test_empty_output(self):
        reasoning, sql = OmniSQLFormatter._parse_output_seq("")
        assert sql == ""


# ============================================================================
# Tests: TrainingSample Conversion
# ============================================================================

class TestTrainingSampleConversion:
    def test_sft_thinking_dict(self):
        sample = TrainingSample(
            db_id="test",
            question="How many heads?",
            schema_context="CREATE TABLE head (id INT);",
            gold_sql="SELECT COUNT(*) FROM head;",
            reasoning="1. Count all rows in head table",
        )
        result = sample.to_sft_thinking_dict()
        assert "instruction" in result
        assert "response" in result
        assert "<think>" in result["response"]
        assert "</think>" in result["response"]
        assert "SELECT COUNT(*) FROM head;" in result["response"]

    def test_sft_dict_no_think_tags(self):
        sample = TrainingSample(
            db_id="test",
            question="How many heads?",
            schema_context="CREATE TABLE head (id INT);",
            gold_sql="SELECT COUNT(*) FROM head;",
        )
        result = sample.to_sft_dict()
        assert "<think>" not in result["response"]
        assert "SELECT COUNT(*) FROM head;" in result["response"]
