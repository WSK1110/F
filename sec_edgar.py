"""
SEC EDGAR API client for automatically fetching 10-K filings by ticker.

Usage:
    filing = fetch_10k("AAPL")
    # filing["local_path"] -> /tmp/.../aapl_10k_2024.pdf
"""
import os
import re
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# Try to import company name helper (graceful if unavailable)
try:
    from RAG_core import extract_company_name as _extract_company
except (ImportError, ModuleNotFoundError):
    # Fallback: simple company name from filename
    import re as _re
    def _extract_company(path):
        name = _re.sub(r'[_-]', ' ', os.path.basename(path))
        name = _re.sub(r'\s+\d{2,4}.*$', '', name)
        return name.strip().title()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SEC_HEADERS = {
    "User-Agent": "DailyProphetRAG/1.0 (contact@dailyprophet.example.com)",
    "Accept": "application/json",
}
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def _cik_padded(cik: int) -> str:
    """Pad CIK to 10 digits."""
    return str(cik).zfill(10)


def lookup_cik(ticker: str) -> Optional[int]:
    """Look up CIK number for a stock ticker."""
    ticker = ticker.upper().strip()
    try:
        req = Request(COMPANY_TICKERS_URL, headers=SEC_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                return int(entry["cik_str"])
        logger.warning(f"Ticker {ticker} not found in SEC database")
        return None
    except Exception as e:
        logger.error(f"CIK lookup failed for {ticker}: {e}")
        return None


def get_latest_10k(cik: int) -> Optional[Dict[str, Any]]:
    """Get the most recent 10-K filing metadata for a company."""
    url = SUBMISSIONS_URL.format(_cik_padded(cik))
    try:
        req = Request(url, headers=SEC_HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        filings = data.get("filings", {}).get("recent", {})
        form_types = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for i, form in enumerate(form_types):
            if form == "10-K":
                acc_no = accession_numbers[i]
                primary_doc = primary_docs[i]
                filing_date = filing_dates[i]
                # Extract year from filing date
                year = filing_date[:4] if filing_date else "unknown"

                return {
                    "accession_number": acc_no,
                    "primary_document": primary_doc,
                    "filing_date": filing_date,
                    "year": year,
                    "cik": cik,
                    # The PDF URL follows this pattern:
                    # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}
                }

        logger.warning(f"No 10-K filing found for CIK {cik}")
        return None
    except Exception as e:
        logger.error(f"Failed to get 10-K info for CIK {cik}: {e}")
        return None


def _acc_no_path(acc_no: str) -> str:
    """Convert accession number to path format (with dashes)."""
    # Remove dashes, re-insert with folder format
    clean = acc_no.replace("-", "")
    return clean


def build_pdf_url(cik: int, accession_number: str, primary_document: str) -> str:
    """Build the URL to download the PDF version of the filing."""
    acc_path = _acc_no_path(accession_number)
    # Try PDF first (most common for 10-K)
    pdf_name = primary_document.rsplit(".", 1)[0] + ".pdf"
    return f"{ARCHIVE_BASE}/{cik}/{acc_path}/{pdf_name}"


def build_html_url(cik: int, accession_number: str, primary_document: str) -> str:
    """Build URL to the HTML/HTM version as fallback."""
    return f"{ARCHIVE_BASE}/{cik}/{_acc_no_path(accession_number)}/{primary_document}"


def download_filing(url: str, output_path: str) -> bool:
    """Download a filing document to a local file."""
    try:
        headers = {**SEC_HEADERS, "Accept": "application/pdf, text/html, */*"}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=60) as resp:
            content = resp.read()
        with open(output_path, "wb") as f:
            f.write(content)
        logger.info(f"Downloaded {len(content)} bytes to {output_path}")
        return True
    except HTTPError as e:
        logger.warning(f"HTTP {e.code} downloading {url}")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def fetch_10k(ticker: str, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Full pipeline: ticker → CIK → 10-K metadata → download PDF.

    Args:
        ticker: Stock ticker (e.g., "AAPL", "MSFT").
        output_dir: Directory to save the PDF. Defaults to /tmp.

    Returns:
        dict with local_path, company, year, or None on failure.
    """
    ticker_upper = ticker.upper().strip()

    # Rate limiting: be polite to SEC
    time.sleep(0.5)

    # Step 1: Look up CIK (with retry)
    cik = None
    for _ in range(2):
        cik = lookup_cik(ticker_upper)
        if cik:
            break
        time.sleep(1)

    if not cik:
        logger.error(f"Could not find CIK for {ticker_upper}")
        return None

    time.sleep(0.3)

    # Step 2: Get latest 10-K metadata (with retry)
    filing = None
    for _ in range(2):
        filing = get_latest_10k(cik)
        if filing:
            break
        time.sleep(1)

    if not filing:
        logger.error(f"Could not find 10-K for {ticker_upper}")
        return None

    time.sleep(0.3)

    # Step 3: Build URLs and download
    # SEC primary documents are usually .htm; try PDF as fallback
    primary_doc = filing["primary_document"]
    html_url = build_html_url(cik, filing["accession_number"], primary_doc)
    output_dir = output_dir or tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    safe_name = ticker_upper.lower().replace(".", "")

    # Check if primary doc ends with .htm/.html
    if primary_doc.lower().endswith(('.htm', '.html')):
        # Download HTM directly
        local_path = os.path.join(output_dir, f"{safe_name}_10k_{filing['year']}.htm")
        downloaded = download_filing(html_url, local_path)
        if not downloaded:
            # Try PDF version
            pdf_url = build_pdf_url(cik, filing["accession_number"], primary_doc)
            local_path = local_path.replace(".htm", ".pdf")
            downloaded = download_filing(pdf_url, local_path)
    else:
        # Primary doc might be PDF
        pdf_url = build_pdf_url(cik, filing["accession_number"], primary_doc)
        local_path = os.path.join(output_dir, f"{safe_name}_10k_{filing['year']}.pdf")
        downloaded = download_filing(pdf_url, local_path)
        if not downloaded:
            local_path = local_path.replace(".pdf", ".htm")
            downloaded = download_filing(html_url, local_path)

    if not downloaded:
        logger.error(f"Failed to download 10-K for {ticker_upper}")
        return None

    # Rate limiting: be polite to SEC
    time.sleep(0.1)

    # Infer company name
    company = _extract_company(local_path) or f"{ticker_upper} Inc."

    return {
        "local_path": local_path,
        "company": company,
        "ticker": ticker_upper,
        "year": filing["year"],
        "filing_date": filing["filing_date"],
        "cik": cik,
    }
