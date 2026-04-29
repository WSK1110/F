"""
SQLite storage for financial tables extracted from 10-K PDFs.
Stores tables in a queryable format for Text-to-SQL.
"""
import os
import sqlite3
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager

from config import CHROMA_DIR

logger = logging.getLogger(__name__)

# SQLite database path (alongside Chroma DB)
DB_DIR = Path(str(CHROMA_DIR)).parent
DB_PATH = DB_DIR / "financial_tables.db"

# ── Schema ──────────────────────────────────────────────────────────────
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    source_file TEXT NOT NULL,
    year INTEGER,
    section TEXT,
    section_name TEXT,
    table_index INTEGER DEFAULT 0,
    caption TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    table_name TEXT NOT NULL,      -- e.g. "Apple_2024_income_statement"
    page_num INTEGER,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS table_columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL,
    col_index INTEGER NOT NULL,
    col_name TEXT NOT NULL,
    col_type TEXT DEFAULT 'TEXT',
    FOREIGN KEY (table_id) REFERENCES tables(id)
);

CREATE TABLE IF NOT EXISTS table_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    row_data TEXT NOT NULL,         -- JSON array of cell values
    FOREIGN KEY (table_id) REFERENCES tables(id)
);
"""


@contextmanager
def _get_db():
    """Get a SQLite connection (context manager)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with _get_db() as conn:
        conn.executescript(CREATE_TABLES_SQL)
    logger.info(f"Table store initialized at {DB_PATH}")


# ── Insert tables ──────────────────────────────────────────────────────
def _sanitize_table_name(name: str) -> str:
    """Create a valid SQL table name from a string."""
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name).lower()
    return name[:60]


def _infer_column_type(values: List[str]) -> str:
    """Infer SQL type from column values."""
    numeric_count = 0
    for v in values:
        v = v.replace(',', '').replace('$', '').replace('%', '').replace('(', '').replace(')', '').strip()
        try:
            float(v)
            numeric_count += 1
        except ValueError:
            pass
    if numeric_count > len(values) * 0.5:
        return 'REAL'
    return 'TEXT'


def store_table(
    company: str,
    source_file: str,
    headers: List[str],
    rows: List[List[str]],
    page_num: int = 0,
    section: Optional[str] = None,
    section_name: Optional[str] = None,
    table_index: int = 0,
    caption: str = "",
    year: Optional[int] = None,
) -> int:
    """Store a parsed table in SQLite.

    Args:
        company: Company name (e.g. "Apple")
        source_file: Original PDF filename
        headers: Column headers (list of strings)
        rows: Data rows (list of lists)
        page_num: Page number in PDF
        section: 10-K section ID
        section_name: Human-readable section name
        table_index: Index of table on the page
        caption: Optional table caption
        year: Fiscal year

    Returns:
        table_id of the stored table, or -1 on failure
    """
    if not headers or not rows:
        return -1

    init_db()

    with _get_db() as conn:
        # Create dataset entry
        cursor = conn.execute(
            """INSERT INTO datasets (company, source_file, year, section, section_name, table_index, caption)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (company, source_file, year, section, section_name, table_index, caption),
        )
        dataset_id = cursor.lastrowid

        # Generate table name
        table_name = _sanitize_table_name(f"{company}_{caption or 'table'}_{table_index}")
        if not table_name:
            table_name = f"table_{dataset_id}"

        # Create table entry
        cursor = conn.execute(
            "INSERT INTO tables (dataset_id, table_name, page_num) VALUES (?, ?, ?)",
            (dataset_id, table_name, page_num),
        )
        table_id = cursor.lastrowid

        # Store column info
        for i, header in enumerate(headers):
            col_values = [row[i] if i < len(row) else "" for row in rows]
            col_type = _infer_column_type(col_values)
            conn.execute(
                "INSERT INTO table_columns (table_id, col_index, col_name, col_type) VALUES (?, ?, ?, ?)",
                (table_id, i, header.strip(), col_type),
            )

        # Store rows
        import json
        for i, row in enumerate(rows):
            # Pad row to match header count
            padded = row + [""] * (len(headers) - len(row))
            conn.execute(
                "INSERT INTO table_rows (table_id, row_index, row_data) VALUES (?, ?, ?)",
                (table_id, i, json.dumps(padded[:len(headers)])),
            )

        logger.info(f"Stored table '{table_name}' ({len(rows)} rows x {len(headers)} cols)")
        return table_id


# ── Query ───────────────────────────────────────────────────────────────
def get_table_schemas() -> List[Dict[str, Any]]:
    """Get all available table schemas for LLM context.

    Returns list of dicts with table_name, columns, row_count, company, etc.
    """
    init_db()
    schemas = []

    with _get_db() as conn:
        tables = conn.execute("""
            SELECT t.id, t.table_name, t.page_num, d.company, d.source_file,
                   d.section_name, d.year, d.caption,
                   (SELECT COUNT(*) FROM table_rows WHERE table_id = t.id) as row_count
            FROM tables t
            JOIN datasets d ON t.dataset_id = d.id
            ORDER BY d.company, d.table_index
        """).fetchall()

        for t in tables:
            cols = conn.execute(
                "SELECT col_name, col_type, col_index FROM table_columns WHERE table_id = ? ORDER BY col_index",
                (t["id"],),
            ).fetchall()

            schemas.append({
                "table_name": t["table_name"],
                "company": t["company"],
                "year": t["year"],
                "section": t["section_name"],
                "caption": t["caption"],
                "columns": [{"name": c["col_name"], "type": c["col_type"]} for c in cols],
                "row_count": t["row_count"],
                "page": t["page_num"],
            })

    return schemas


def execute_sql(sql: str) -> Dict[str, Any]:
    """Execute a SQL query against the table store.

    Args:
        sql: A SELECT SQL statement.

    Returns:
        dict with columns, rows, error (if any).
    """
    # Security: only allow SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed", "columns": [], "rows": []}

    init_db()

    try:
        with _get_db() as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = [list(row) for row in cursor.fetchall()]

        return {"columns": columns, "rows": rows, "error": None}
    except sqlite3.Error as e:
        logger.warning(f"SQL execution error: {e}")
        return {"error": str(e), "columns": [], "rows": []}


def get_table_context(table_name: str, max_rows: int = 20) -> Optional[str]:
    """Get a human-readable representation of a table for LLM context."""
    init_db()

    with _get_db() as conn:
        table = conn.execute(
            "SELECT t.id, d.company, d.year, d.caption FROM tables t JOIN datasets d ON t.dataset_id = d.id WHERE t.table_name = ?",
            (table_name,),
        ).fetchone()

        if not table:
            return None

        cols = conn.execute(
            "SELECT col_name FROM table_columns WHERE table_id = ? ORDER BY col_index",
            (table["id"],),
        ).fetchall()

        rows_data = conn.execute(
            "SELECT row_data FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT ?",
            (table["id"], max_rows),
        ).fetchall()

    import json
    header = " | ".join(c["col_name"] for c in cols)
    separator = "-" * len(header)
    lines = [
        f"Table: {table_name}",
        f"Company: {table['company']}  |  Year: {table['year']}  |  Caption: {table['caption']}",
        "",
        header,
        separator,
    ]

    for row in rows_data:
        cells = json.loads(row["row_data"])
        lines.append(" | ".join(str(c) for c in cells))

    return "\n".join(lines)


def clear_tables():
    """Clear all table data (for re-ingestion)."""
    with _get_db() as conn:
        conn.execute("DELETE FROM table_rows")
        conn.execute("DELETE FROM table_columns")
        conn.execute("DELETE FROM tables")
        conn.execute("DELETE FROM datasets")
    logger.info("All table data cleared")


def count_tables() -> int:
    """Return the number of stored tables."""
    init_db()
    with _get_db() as conn:
        return conn.execute("SELECT COUNT(*) as cnt FROM tables").fetchone()["cnt"]
