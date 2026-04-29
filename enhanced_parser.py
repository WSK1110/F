"""
Enhanced document parser with table-aware PDF extraction.
Uses pdfplumber for precise table and text extraction from financial reports.
"""
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from langchain.schema import Document

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not installed; falling back to PyPDFLoader")

# Lazy import for table store (avoids circular dep)
_table_store = None
def _get_table_store():
    global _table_store
    if _table_store is None:
        import table_store as ts
        _table_store = ts
    return _table_store


@dataclass
class TableData:
    """Structured table extracted from a PDF page."""
    page_num: int
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""


# ── 10-K Section patterns ─────────────────────────────────────────────────
SEC_10K_PATTERNS = {
    "item_1": r"item\s*1[\.\s]*business",
    "item_1a": r"item\s*1a[\.\s]*risk\s*factors",
    "item_1b": r"item\s*1b[\.\s]*unresolved\s*staff\s*comments",
    "item_2": r"item\s*2[\.\s]*properties",
    "item_3": r"item\s*3[\.\s]*legal\s*proceedings",
    "item_4": r"item\s*4[\.\s]*mine\s*safety",
    "item_5": r"item\s*5[\.\s]*market\s*for\s*registrant'?s?\s*common\s*equity",
    "item_6": r"item\s*6[\.\s]*selected\s*financial\s*data",
    "item_7": r"item\s*7[\.\s]*management'?s?\s*discussion",
    "item_7a": r"item\s*7a[\.\s]*quantitative\s*and\s*qualitative",
    "item_8": r"item\s*8[\.\s]*financial\s*statements",
    "item_9": r"item\s*9[\.\s]*changes\s*in\s*and\s*disagreements",
    "item_9a": r"item\s*9a[\.\s]*controls\s*and\s*procedures",
    "item_9b": r"item\s*9b[\.\s]*other\s*information",
    "item_10": r"item\s*10[\.\s]*directors",
    "item_11": r"item\s*11[\.\s]*executive\s*compensation",
    "item_12": r"item\s*12[\.\s]*security\s*ownership",
    "item_13": r"item\s*13[\.\s]*certain\s*relationships",
    "item_14": r"item\s*14[\.\s]*principal\s*accountant",
    "item_15": r"item\s*15[\.\s]*exhibits",
}

# Reverse mapping: section_id -> display name
SECTION_NAMES = {
    "item_1": "Business",
    "item_1a": "Risk Factors",
    "item_1b": "Unresolved Staff Comments",
    "item_2": "Properties",
    "item_3": "Legal Proceedings",
    "item_4": "Mine Safety Disclosures",
    "item_5": "Market for Common Equity",
    "item_6": "Selected Financial Data",
    "item_7": "Management's Discussion & Analysis (MD&A)",
    "item_7a": "Quantitative & Qualitative Disclosures",
    "item_8": "Financial Statements",
    "item_9": "Changes in Accounting",
    "item_9a": "Controls and Procedures",
    "item_9b": "Other Information",
    "item_10": "Directors and Governance",
    "item_11": "Executive Compensation",
    "item_12": "Security Ownership",
    "item_13": "Certain Relationships",
    "item_14": "Principal Accountant Fees",
    "item_15": "Exhibits",
}


def extract_year_from_text(text: str) -> Optional[int]:
    """Extract fiscal year from text (e.g., 'Fiscal Year Ended December 31, 2024')."""
    patterns = [
        r"fiscal\s+year\s+ended\s+\w+\s+\d+,\s*(\d{4})",
        r"for\s+the\s+(?:fiscal\s+)?year\s+(?:ended\s+)?(\d{4})",
        r"annual\s+report.*?(\d{4})",
        r"(\d{4})\s+(?:annual\s+)?(?:report|10[-\s]?k)",
        r"december\s+31,\s*(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            if 1990 <= year <= 2030:
                return year
    return None


def detect_10k_section(text: str, page_num: int) -> Optional[str]:
    """Detect which 10-K Item section the text belongs to."""
    for section_id, pattern in SEC_10K_PATTERNS.items():
        if re.search(pattern, text[:500], re.IGNORECASE):
            return section_id

    # Fallback: check first 200 chars for "Item X" patterns
    m = re.match(r"item\s+(\d+[a-z]?)", text[:200].strip().lower())
    if m:
        key = f"item_{m.group(1)}"
        if key in SEC_10K_PATTERNS:
            return key

    return None


def pdfplumber_load(file_path: str) -> List[Document]:
    """Load PDF with table-aware extraction using pdfplumber.

    Returns a mix of text Documents and table-as-text Documents, each tagged
    with page number, company name, 10-K section, and year metadata.
    """
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required for enhanced parsing")

    from RAG_core import extract_company_name

    documents = []
    company = extract_company_name(file_path)
    full_text = ""

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            full_text += text

            # Extract tables
            tables = page.extract_tables()
            table_texts = []

            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                rows = []
                for row in table:
                    cleaned = [c.replace("\n", " ").strip() if c else "" for c in row]
                    rows.append(cleaned)

                # First row is header
                header = rows[0] if rows else []
                data_rows = rows[1:] if len(rows) > 1 else []

                # Skip empty tables
                if not data_rows:
                    continue

                # Convert table to structured text
                lines = [f"[Table {table_idx + 1}]"]
                if header:
                    lines.append(" | ".join(header))
                    lines.append("-" * len(" | ".join(header)))
                for data_row in data_rows:
                    lines.append(" | ".join(data_row))

                table_text = "\n".join(lines)
                table_texts.append(table_text)

                # Store table in SQLite for Text-to-SQL
                try:
                    ts = _get_table_store()
                    ts.store_table(
                        company=company,
                        source_file=file_path,
                        headers=header,
                        rows=data_rows,
                        page_num=page_num,
                        section=None,  # detected later
                        section_name=None,
                        table_index=table_idx,
                        caption="",
                        year=None,  # detected later
                    )
                except Exception as e:
                    logger.debug(f"Failed to store table in SQLite: {e}")

                # Create a separate document for significant tables
                if len(data_rows) >= 3:
                    table_doc = Document(
                        page_content=table_text,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "company": company,
                            "type": "table",
                            "rows": len(data_rows),
                            "columns": len(header) if header else 0,
                        }
                    )
                    documents.append(table_doc)

            # Mark table positions in text
            if table_texts:
                text += "\n\n[Extracted Tables on this page]\n" + "\n\n".join(table_texts)

            if text.strip():
                text_doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        "company": company,
                        "type": "text",
                    }
                )
                documents.append(text_doc)

    # Detect year from full text
    year = extract_year_from_text(full_text)
    if year:
        for doc in documents:
            doc.metadata["year"] = year

    # Detect 10-K sections and tag documents
    _tag_sections(documents)

    logger.info(f"pdfplumber loaded {file_path}: {len(documents)} segments "
                f"({sum(1 for d in documents if d.metadata.get('type') == 'table')} tables)")
    return documents


def _tag_sections(documents: List[Document]):
    """Tag documents with their 10-K section based on content."""
    current_section = None
    for doc in documents:
        detected = detect_10k_section(doc.page_content, doc.metadata.get("page", 0))
        if detected:
            current_section = detected
        doc.metadata["section"] = current_section
        doc.metadata["section_name"] = SECTION_NAMES.get(current_section, "General") if current_section else "General"
