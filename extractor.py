"""
extractor.py - Invoice data extraction engine.

Two-path extraction strategy:
1. PDF with native text layer (≥ 50 chars via PyMuPDF page.get_text()):
   → Use that text directly. No OCR needed. Much faster and more accurate.
2. Scanned images or PDFs with empty/sparse text layers:
   → Fall back to preprocessing (deskew, threshold) + Tesseract OCR.

Supported formats for regex parsing:
- Invoice labels: "Invoice No", "Invoice Number", "Bill of Supply Number",
  "Invoice #" (case-insensitive)
- Date formats: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, "10 Aug 2026"
  Labels: "Invoice Date", "Bill of Supply Date"
- Vendor: from "Billed From" / "Sold By" block. Falls back to INVOICE header.
  Never uses document titles like "Bill of Supply" or "Tax Invoice".
- Currency: accepts ₹, Rs, INR, $ and comma-thousand separators.
  Handles OCR slips (O/0, S/5, Z/2, a/4, b/8) for scanned documents.
- Totals: uses the LAST currency token on the "Total" row.
  Tax = sum of SGST, CGST, IGST columns.
  Subtotal = "Taxable Value" column if present, else Gross.
- Line items: from "Particulars"/"Product"/"Description" table header
  to "Total" row. Skips "CGST 0.0 %" / "SGST 0.0 %" rate lines.
- Tolerance: subtotal + tax == total passes when tax is 0.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pymupdf


# ---------------------------------------------------------------------------
# Currency parsing
# ---------------------------------------------------------------------------

def parse_currency(val_str: Optional[str]) -> Optional[float]:
    """
    Parse a monetary token to float.

    Handles:
    - Currency symbols: ₹, Rs, INR, $, USD
    - Comma thousand-separators: 1,234.50 → 1234.50
    - Space-separated cents: "1,100 00" → 1100.00
    - OCR letter-to-digit slips: O→0, S→5, Z→2, l→1, a→4, b→8
    - Trailing OCR noise
    """
    if not val_str:
        return None
    s = str(val_str).strip()

    # Remove currency symbols and words (₹, Rs., INR, $, USD)
    s = re.sub(r"[₹\$\u20b9]|Rs\.?|INR|USD", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("|", "").replace("~", "").strip()

    # Space-separated cents: "1,100 00" → "1,100.00"
    m_space = re.search(r"([0-9,.]+)\s+([0-9Oo]{2})$", s)
    if m_space:
        s = m_space.group(1) + "." + m_space.group(2)

    # OCR letter-to-digit substitutions (only applied for scanned docs,
    # but harmless for native PDFs since they won't contain these)
    rep = {
        "O": "0", "o": "0",
        "S": "5", "s": "5",
        "Z": "2", "z": "2",
        "l": "1", "I": "1",
        "a": "4", "A": "4",
        "b": "8", "B": "8",
    }
    cleaned = "".join(rep.get(c, c) for c in s)
    cleaned = re.sub(r"[^\d.,]", "", cleaned)
    if not cleaned:
        return None

    # Normalise decimal/comma separators
    if "." in cleaned and "," in cleaned:
        cleaned = cleaned.replace(",", "")               # 1,234.50 → 1234.50
    elif cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]  # 1.234.50 → 1234.50
    elif cleaned.count(",") > 1:
        parts = cleaned.split(",")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]  # 1,234,50 → 1234.50
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts[1]) == 2:
            cleaned = parts[0] + "." + parts[1]          # 1,50 → 1.50 (French)
        else:
            cleaned = cleaned.replace(",", "")            # 1,234 → 1234

    # 4-digit trailing-zero collapse: "7500" → "75.00"  (only for scanned OCR)
    if "." not in cleaned and len(cleaned) >= 4 and cleaned.endswith("00"):
        cleaned = cleaned[:-2] + ".00"

    try:
        return float(cleaned)
    except ValueError:
        return None


def _has_currency_symbol(tok: str) -> bool:
    """Return True if token contains any currency symbol or prefix."""
    return any(sym in tok for sym in ["₹", "$", "\u20b9", "Rs", "rs", "INR", "USD"])


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_line(line: str) -> str:
    """Fix common OCR noise patterns (for scanned-image path only)."""
    norm = re.sub(r"([$]?[0-9a-zA-Z]{1,4})\s+([0-9Oo]{2})\b", r"\1.\2", line)
    norm = re.sub(r"\$\$([0-9])", r"$5\1", norm)
    norm = re.sub(r"(?<![\d,])4(\d{2,3}\.\d{2})\b", r"$\1", norm)
    return norm


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def extract_vendor(lines: List[str]) -> Optional[str]:
    """
    Extract vendor name.
    Priority:
      1. Text after "Billed From" or "Sold By" label (first non-empty line)
      2. Text before "INVOICE" on the same header line
      3. First non-title line
    """
    BAD_TITLES = ["bill of supply", "tax invoice", "invoice", "proforma"]

    # Priority 1: Billed From / Sold By
    for i, l in enumerate(lines):
        if re.search(r"\b(?:Billed\s*From|Sold\s*By)\b", l, re.I):
            # The vendor may be on the same line or next line
            # Split by 2+ spaces (multi-column layout)
            rest = re.sub(r"\b(?:Billed\s*From|Sold\s*By)\b", "", l, flags=re.I).strip()
            if rest:
                cols = re.split(r"\s{2,}", rest)
                v = cols[0].strip()
                if v and not any(t in v.lower() for t in BAD_TITLES) and len(v) > 3:
                    return v
            # Check next line
            if i + 1 < len(lines):
                cand = lines[i + 1]
                cols = re.split(r"\s{2,}", cand)
                v = cols[0].strip()
                if v and not any(t in v.lower() for t in BAD_TITLES) and len(v) > 3:
                    return v

    # Priority 2: Before "INVOICE" on header
    for l in lines[:6]:
        if "invoice" in l.lower():
            v_cand = re.sub(r"\s+INVOICE.*$", "", l, flags=re.I).strip()
            v_cand = re.sub(r"^[^\w]+", "", v_cand)
            if len(v_cand) > 3 and not any(t in v_cand.lower() for t in BAD_TITLES):
                return v_cand

    # Priority 3: First non-title line
    for l in lines[:3]:
        cand = re.sub(r"^[^\w]+", "", l).strip()
        if cand and not any(t in cand.lower() for t in BAD_TITLES) and len(cand) > 3:
            return cand

    return None


def extract_invoice_number(text: str) -> Optional[str]:
    """
    Accept labels: Invoice No, Invoice Number, Bill of Supply Number, Invoice #
    Takes the alphanumeric token after the colon.
    """
    # Explicit labels
    m = re.search(
        r"(?:Invoice\s*(?:No\.?|Number|#)|Bill\s*of\s*Supply\s*Number)\s*[:\-–]?\s*([A-Za-z0-9][A-Za-z0-9\-_./]{2,})",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # Generic INV-XXXX-XXXX pattern
    m2 = re.search(r"\bINV[-_.\s]*([0-9]{4})[-_.\s]*([0-9]{4})\b", text, re.IGNORECASE)
    if m2:
        return f"INV-{m2.group(1)}-{m2.group(2)}"

    # Reference fallback
    m3 = re.search(r"(?:Invoice|Invorce|Reference)\s*[:\-–]?\s*([A-Z0-9][A-Z0-9\-_.]{3,})", text, re.IGNORECASE)
    if m3:
        return m3.group(1).strip()

    return None


_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def normalise_date(raw: str) -> str:
    """Convert any matched date to DD-MM-YYYY where possible."""
    raw = raw.strip()
    # YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"([0-9]{4})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})", raw)
    if m:
        return f"{m.group(3).zfill(2)}-{m.group(2).zfill(2)}-{m.group(1)}"
    # DD-MM-YYYY, DD/MM/YYYY
    m = re.match(r"([0-9]{1,2})[-/.]([0-9]{1,2})[-/.]([0-9]{4})", raw)
    if m:
        return f"{m.group(1).zfill(2)}-{m.group(2).zfill(2)}-{m.group(3)}"
    # "10 Aug 2026"
    m = re.match(r"([0-9]{1,2})\s+([A-Za-z]{3,9})\s+([0-9]{4})", raw)
    if m:
        month_num = _MONTH_MAP.get(m.group(2).lower()[:3])
        if month_num:
            return f"{m.group(1).zfill(2)}-{month_num}-{m.group(3)}"
    return raw  # return as-is if no match


def extract_date(text: str) -> Optional[str]:
    """
    Accept labels: Invoice Date, Bill of Supply Date.
    Accept formats: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, "10 Aug 2026".
    """
    DATE_PAT = (
        r"([0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4}"
        r"|[0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2}"
        r"|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})"
    )
    # Labelled
    m = re.search(
        r"(?:Invoice\s*Date|Bill\s*of\s*Supply\s*Date)\s*[:\-–]?\s*" + DATE_PAT,
        text, re.IGNORECASE
    )
    if m:
        return normalise_date(m.group(1))

    # Any date
    m2 = re.search(DATE_PAT, text)
    if m2:
        return normalise_date(m2.group(1))

    return None


# ---------------------------------------------------------------------------
# Table-aware totals + line-item extraction
# ---------------------------------------------------------------------------

def _extract_via_table(lines: List[str]) -> Tuple[
    Optional[float], Optional[float], Optional[float], List[Dict[str, Any]]
]:
    """
    Parse a structured GST-style table:
      Header row contains Particulars/Product/Description
      Total row contains the grand total
    Returns (subtotal, tax, total, line_items).
    """
    table_hdr_idx = -1
    total_row_idx = -1

    for i, l in enumerate(lines):
        if re.search(r"\b(?:Particulars|Product|Description)\b", l, re.I) and table_hdr_idx == -1:
            table_hdr_idx = i
        if re.search(r"^\s*Total\b", l, re.I) and i > table_hdr_idx and total_row_idx == -1:
            total_row_idx = i

    if table_hdr_idx == -1 or total_row_idx == -1:
        return None, None, None, []

    hdr_cols = lines[table_hdr_idx].lower().split()
    total_line = lines[total_row_idx]
    tot_tokens = total_line.split()

    # Gather currency amounts from Total row
    curr_amounts: List[float] = []
    for tok in tot_tokens:
        if tok.lower() == "total":
            continue
        v = parse_currency(tok)
        if v is None:
            continue
        if _has_currency_symbol(tok):
            curr_amounts.append(v)
        elif ("." in tok and len(tok.split(".")[-1]) == 2 and curr_amounts):
            # Decimal token adjacent to currency values (sorted-text PDFs)
            curr_amounts.append(v)

    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    if curr_amounts:
        total = curr_amounts[-1]

    # GST format: [Gross, Taxable, SGST, CGST, Total] => 4+ currency amounts
    if len(curr_amounts) >= 4:
        # Subtotal = Taxable Value column
        has_taxable = any("taxable" in c for c in hdr_cols)
        subtotal = curr_amounts[1] if has_taxable else curr_amounts[0]
        # Tax = sum of GST columns between Taxable and Total
        tax = round(sum(curr_amounts[2:-1]), 2)
    elif len(curr_amounts) == 1:
        subtotal = curr_amounts[0]

    # Line items
    line_items: List[Dict[str, Any]] = []
    for r_i in range(table_hdr_idx + 1, total_row_idx):
        row_l = lines[r_i]
        # Skip GST rate lines like "CGST 0.0 %" or "SGST 0.0 %"
        if re.search(r"\b(?:CGST|SGST|IGST)\s+[\d.]+\s*%", row_l, re.I):
            continue
        r_tokens = row_l.split()
        if len(r_tokens) < 2:
            continue

        r_curr: List[float] = []
        for tok in r_tokens:
            if _has_currency_symbol(tok):
                v = parse_currency(tok)
                if v is not None:
                    r_curr.append(v)

        if not r_curr:
            continue

        # Description: tokens before SAC code (6+ digit number) or float or currency symbol
        desc_parts: List[str] = []
        for tok in r_tokens:
            if any(s in tok for s in ["₹", "$", "\u20b9"]):
                break
            if tok.isdigit() and len(tok) >= 4:  # SAC / HSN code
                break
            try:
                float(tok)
                break
            except ValueError:
                desc_parts.append(tok)
        desc = " ".join(desc_parts).strip() or "Item"

        # Qty: first float-looking token < 1000 (SAC codes are 6+ digits, skip them)
        qty = 1
        for tok in r_tokens:
            try:
                q_f = float(tok)
                if 0 < q_f < 1000:
                    qty = int(q_f) if q_f.is_integer() else q_f
                    break
            except ValueError:
                pass

        amt = r_curr[-1]
        price = round(amt / qty, 2) if qty > 0 else amt
        line_items.append({
            "description": desc,
            "qty": qty,
            "unit_price": price,
            "amount": round(amt, 2),
        })

    return subtotal, tax, total, line_items


def _extract_via_regex(lines: List[str], text: str) -> Tuple[
    Optional[float], Optional[float], Optional[float], List[Dict[str, Any]]
]:
    """
    Fallback regex-based extractor for OCR text from scanned invoices.
    Handles common OCR slips. Matches Subtotal, Tax, Total rows.
    """
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    for l in lines:
        ln = normalize_line(l)
        # Subtotal
        if re.search(r"(?:Subtotal|Subtotas|Sub[\s\-_]*Total)", ln, re.I) and subtotal is None:
            subtotal = parse_currency(
                re.sub(r"^.*?(?:Subtotal|Subtotas|Sub[\s\-_]*Total)", "", ln, flags=re.I)
            )
        # Tax (SGST / CGST / IGST sum, or single Tax row)
        elif re.search(r"\b(?:Tax|Tan|VAT)\b", ln, re.I) and tax is None:
            no_pct = re.sub(r"\([^\)]*\)", "", ln)
            val = parse_currency(re.sub(r"^.*?(?:Tax|Tan|VAT)", "", no_pct, flags=re.I))
            if val is not None:
                if subtotal and val > subtotal * 0.3:
                    s_t = str(val)
                    if len(s_t) > 4 and s_t[0] in "34":
                        val = float(s_t[1:])
                tax = val
        # Total / Total Amount Due
        elif re.search(r"(?:Total\s*Amount\s*Due|Grand\s*Total|(?<!Sub)Total)", ln, re.I) and total is None:
            m = re.search(
                r"(?:Total\s*Amount\s*Due|Grand\s*Total|(?<!Sub)Total)[^\d$₹]*([$₹0-9OoSszZlI.,\s]+)",
                ln, re.I
            )
            if m:
                total = parse_currency(m.group(1))

    # Line items
    line_items: List[Dict[str, Any]] = []
    start_idx = -1
    end_idx = -1
    for i, l in enumerate(lines):
        if any(w in l.lower() for w in ["currency", "usd", "account reference", "recosnl", "reterence", "resesence"]):
            start_idx = i
        if any(w in l.lower() for w in ["subtotal", "subtotas"]) and end_idx == -1:
            end_idx = i

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        for raw in lines[start_idx + 1:end_idx]:
            if any(w in raw.lower() for w in ["description", "unit price", "amount", "funrr", "pamoun"]):
                continue
            norm = normalize_line(raw)
            tokens = norm.split()
            if len(tokens) < 3:
                continue

            raw_amt = tokens[-1]
            raw_price = tokens[-2]
            raw_qty = tokens[-3] if len(tokens) >= 4 else "1"

            amt = parse_currency(raw_amt)
            price = parse_currency(raw_price)
            if not amt and not price:
                continue

            q_clean = raw_qty.replace("z", "2").replace("Z", "2").replace("a", "4").replace("?", "2").replace("o", "0")
            q_digits = re.findall(r"\d+", q_clean)
            qty = int(q_digits[0]) if q_digits else 1

            desc = " ".join(tokens[:-3] if len(tokens) >= 4 else tokens[:-2])
            desc = re.sub(r"^[^\w]+", "", desc).strip()

            if amt and qty and not price:
                price = round(amt / qty, 2)
            elif price and qty and not amt:
                amt = round(qty * price, 2)
            elif price and amt and qty:
                expected = round(qty * price, 2)
                if abs(expected - amt) > 1.0:
                    calc_price = round(amt / qty, 2)
                    if abs(calc_price * qty - amt) < 0.05:
                        price = calc_price
                    else:
                        amt = expected

            line_items.append({
                "description": desc,
                "qty": qty,
                "unit_price": price,
                "amount": amt,
            })

    return subtotal, tax, total, line_items


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_invoice_data(
    ocr_text: str,
    pdf_native_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main extraction function.

    Args:
        ocr_text:         Text from Tesseract OCR (always provided).
        pdf_native_text:  Text from PyMuPDF page.get_text() if the input was a PDF.
                          When this is ≥ 50 chars, it is used instead of OCR text.

    Returns dict with keys:
        vendor, invoice_number, date, line_items, subtotal, tax, total
        (any field that cannot be found is None)
    """
    # Choose text source
    use_native = pdf_native_text is not None and len(pdf_native_text.strip()) >= 50
    working_text = pdf_native_text if use_native else ocr_text

    lines = [l.strip() for l in working_text.splitlines() if l.strip()]

    vendor = extract_vendor(lines)
    inv_num = extract_invoice_number(working_text)
    date = extract_date(working_text)

    # Try structured table first, fall back to regex
    subtotal, tax, total, line_items = _extract_via_table(lines)
    if subtotal is None and total is None:
        subtotal, tax, total, line_items = _extract_via_regex(lines, working_text)

    # If tax is still None but subtotal ≈ total, set tax = 0.0
    if tax is None and subtotal is not None and total is not None:
        if abs(subtotal - total) <= 0.01:
            tax = 0.0

    return {
        "vendor": vendor,
        "invoice_number": inv_num,
        "date": date,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
