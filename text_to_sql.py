"""
Text-to-SQL engine for financial table queries.

Converts natural language questions into SQL queries against the table store,
executes them, and formats the results. Uses the configured LLM for SQL generation.
"""
import json
import re
import logging
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

from table_store import get_table_schemas, execute_sql, get_table_context

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial SQL expert. Given a question about financial tables and the available table schemas, generate a SQLite SELECT query to answer the question.

RULES:
1. Only generate SELECT queries (no INSERT/UPDATE/DELETE/CREATE/DROP).
2. Use the exact column names from the schema.
3. Table names are case-sensitive — match them exactly as shown.
4. Column values might contain special characters ($, %, commas) — use REPLACE() or CAST() for numeric comparisons.
5. For percentage values, strip '%' before comparing.
6. For currency values, strip '$' and commas before comparing.
7. The column `row_data` in table_rows stores cell values as a JSON array — to access individual columns use json_extract(row_data, '$[0]'), json_extract(row_data, '$[1]'), etc.
8. Join tables with table_rows using tables.id = table_rows.table_id.
9. Return only the SQL query, no markdown formatting, no explanation.
10. If the question cannot be answered with available tables, respond with: -- NO_ANSWER: [reason]"""


def _build_schema_context() -> str:
    """Build a string describing all available table schemas."""
    schemas = get_table_schemas()
    if not schemas:
        return "No financial tables available."

    parts = ["Available tables (schema below):\n"]
    for s in schemas:
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in s['columns'])
        parts.append(
            f"Table: {s['table_name']}\n"
            f"  Company: {s['company']}\n"
            f"  Year: {s.get('year', 'N/A')}\n"
            f"  Section: {s.get('section', 'N/A')}\n"
            f"  Caption: {s.get('caption', 'N/A')}\n"
            f"  Columns ({len(s['columns'])}): {cols}\n"
            f"  Rows: {s['row_count']}\n"
        )

    return "\n".join(parts)


def generate_sql(question: str, llm) -> Optional[str]:
    """Use LLM to generate a SQL query from a natural language question.

    Args:
        question: Natural language question.
        llm: LangChain LLM instance.

    Returns:
        SQL query string, or None if the question can't be answered.
    """
    schema_context = _build_schema_context()

    if "No financial tables available" in schema_context:
        logger.warning("No tables available for Text-to-SQL")
        return None

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Available tables:\n{schema_context}\n\nQuestion: {question}\n\nSQL:"),
    ]

    try:
        result = llm(messages)
        sql = result.content if hasattr(result, 'content') else str(result)
        sql = sql.strip()

        # Check for NO_ANSWER
        if sql.startswith("-- NO_ANSWER"):
            logger.info(f"LLM could not answer: {sql}")
            return None

        # Clean up markdown code fences if present
        sql = re.sub(r'^```(?:sql)?\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        sql = sql.strip()

        # Validate it's a SELECT
        if not sql.upper().strip().startswith("SELECT"):
            logger.warning(f"Generated non-SELECT statement: {sql[:100]}")
            return None

        logger.info(f"Generated SQL: {sql[:200]}...")
        return sql

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return None


def explain_question(question: str, llm) -> Optional[str]:
    """Generate a SQL query and return a natural language explanation
    of what data the user is asking for (without executing)."""
    sql = generate_sql(question, llm)
    if not sql:
        return "Cannot determine what tables or data would answer this question. Try rephrasing."

    try:
        messages = [
            SystemMessage(content="Explain what the following SQL query does in plain financial English, in 1-2 sentences."),
            HumanMessage(content=f"Question: {question}\nSQL: {sql}\n\nExplanation:"),
        ]
        result = llm(messages)
        explanation = result.content if hasattr(result, 'content') else str(result)
        return explanation.strip()
    except Exception as e:
        logger.warning(f"Explain failed: {e}")
        return f"Query would use: {sql[:150]}..."


def ask_table(question: str, llm) -> Dict[str, Any]:
    """Full Text-to-SQL pipeline: question → SQL → execute → format result.

    Args:
        question: Natural language question about financial tables.
        llm: LangChain LLM instance.

    Returns:
        dict with:
        - sql: the generated SQL
        - columns: result column names
        - rows: result data rows
        - error: error message if any
        - answer: formatted natural language answer
        - has_data: whether results contain data
    """
    # Generate SQL
    sql = generate_sql(question, llm)
    if not sql:
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "error": "Could not generate a query from this question. Tables may be unavailable or question is out of scope.",
            "answer": "",
            "has_data": False,
        }

    # Execute
    result = execute_sql(sql)

    if result.get("error"):
        return {
            "sql": sql,
            "columns": [],
            "rows": [],
            "error": f"SQL execution error: {result['error']}",
            "answer": "",
            "has_data": False,
        }

    # Format result
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    has_data = len(rows) > 0

    # Build natural language answer
    answer = _format_result(question, sql, columns, rows, llm)

    return {
        "sql": sql,
        "columns": columns,
        "rows": rows[:50],  # Cap at 50 rows
        "error": None,
        "answer": answer,
        "has_data": has_data,
        "total_rows": len(rows),
    }


def _format_result(question: str, sql: str, columns: List[str], rows: List[List], llm) -> str:
    """Format SQL query results as natural language."""
    if not rows:
        return "The query returned no results for this question."

    # For single numeric results, format directly
    if len(columns) == 1 and len(rows) == 1:
        val = rows[0][0]
        return f"**{columns[0]}**: {val}"

    # For simple results, build a markdown summary
    if len(rows) <= 10:
        try:
            messages = [
                SystemMessage(content="Summarize the following data query result as a concise financial answer. Use specific numbers. Max 3 sentences."),
                HumanMessage(content=f"Question: {question}\nSQL: {sql}\n\nColumns: {columns}\nRows: {json.dumps(rows[:5])}\n\nAnswer:"),
            ]
            result = llm(messages)
            summary = result.content if hasattr(result, 'content') else str(result)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Result formatting failed: {e}")

    # Fallback: table format
    lines = [f"Query returned {len(rows)} rows."]
    lines.append(" | ".join(columns))
    lines.append("-" * len(" | ".join(columns)))
    for row in rows[:5]:
        lines.append(" | ".join(str(c) for c in row))
    if len(rows) > 5:
        lines.append(f"... and {len(rows) - 5} more rows")
    return "\n".join(lines)


# ── Detect table question ───────────────────────────────────────────────
TABLE_KEYWORDS = [
    "revenue", "income", "profit", "loss", "balance sheet", "income statement",
    "cash flow", "financial statement", "earnings", "margin", "ratio",
    "quarter", "fiscal", "year-over-year", "yoy", "qoq",
    "increase", "decrease", "growth", "decline", "compare",
    "how much", "what was", "what were", "what is", "what are",
    "total", "net", "gross", "operating",
    "table", "chart", "figure", "schedule",
]


def is_table_question(question: str) -> bool:
    """Heuristic: is this question likely answerable from tabular data?"""
    q_lower = question.lower()

    # Direct table indicators
    if any(q_lower.startswith(kw) for kw in ["show", "list", "display"]):
        return True

    # Has financial keywords + asks for specific numbers
    has_financial = any(kw in q_lower for kw in TABLE_KEYWORDS)
    has_number_word = any(w in q_lower for w in ["how much", "how many", "what", "which", "compare"])
    has_number_pattern = bool(re.search(r'\d{4}', q_lower))  # year reference

    return (has_financial and has_number_word) or has_number_pattern
