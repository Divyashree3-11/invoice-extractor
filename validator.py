"""
validator.py - Arithmetic and business logic validation for invoice extraction.

Validates extracted invoice data:
1. Missing fields check: ensures vendor, invoice number, date, subtotal, total, and line items exist.
2. Line items sum check: sum(line item amounts) == subtotal (tolerance 0.01).
3. Grand total check: subtotal + tax == total (tolerance 0.01).

Returns a list of dicts:
[
    {"check": str, "status": "PASS" | "FAIL", "message": str},
    ...
]
"""

from typing import Dict, Any, List, Optional


def validate_invoice(invoice_data: Dict[str, Any], tolerance: float = 0.01) -> List[Dict[str, str]]:
    """
    Validates extracted invoice dictionary and returns a list of verification checks.

    Args:
        invoice_data: Dictionary returned by extractor.py with keys:
                      vendor, invoice_number, date, line_items, subtotal, tax, total.
        tolerance: Allowed floating point discrepancy (default: 0.01).

    Returns:
        List of dicts with keys: 'check', 'status', 'message'.
    """
    results: List[Dict[str, str]] = []

    # -------------------------------------------------------------
    # 1. Missing Fields Check
    # -------------------------------------------------------------
    required_fields = ["vendor", "invoice_number", "date", "subtotal", "total"]
    missing: List[str] = [f for f in required_fields if invoice_data.get(f) is None]

    items = invoice_data.get("line_items")
    if not items or len(items) == 0:
        missing.append("line_items")

    if missing:
        results.append({
            "check": "Required Fields",
            "status": "FAIL",
            "message": f"Missing critical field(s): {', '.join(missing)}.",
        })
    else:
        results.append({
            "check": "Required Fields",
            "status": "PASS",
            "message": "All required fields (vendor, invoice #, date, totals, line items) are present.",
        })

    # -------------------------------------------------------------
    # 2. Line Items Sum == Subtotal Check
    # -------------------------------------------------------------
    subtotal: Optional[float] = invoice_data.get("subtotal")

    if items and len(items) > 0 and subtotal is not None:
        calc_items_sum = sum((it.get("amount") or 0.0) for it in items)
        diff = round(calc_items_sum - subtotal, 2)
        if abs(diff) <= tolerance:
            results.append({
                "check": "Line Items Sum vs Subtotal",
                "status": "PASS",
                "message": f"Sum of {len(items)} line item(s) (${calc_items_sum:,.2f}) matches stated subtotal (${subtotal:,.2f}).",
            })
        else:
            results.append({
                "check": "Line Items Sum vs Subtotal",
                "status": "FAIL",
                "message": (
                    f"Line items sum (${calc_items_sum:,.2f}) differs from subtotal (${subtotal:,.2f}) "
                    f"by ${diff:+,.2f}."
                ),
            })
    else:
        results.append({
            "check": "Line Items Sum vs Subtotal",
            "status": "FAIL",
            "message": "Cannot reconcile line items sum: Subtotal or line items are missing.",
        })

    # -------------------------------------------------------------
    # 3. Subtotal + Tax == Total Check
    # -------------------------------------------------------------
    total: Optional[float] = invoice_data.get("total")
    tax: float = invoice_data.get("tax") or 0.0

    if subtotal is not None and total is not None:
        expected_total = round(subtotal + tax, 2)
        total_diff = round(total - expected_total, 2)
        if abs(total_diff) <= tolerance:
            results.append({
                "check": "Subtotal + Tax == Total",
                "status": "PASS",
                "message": f"Subtotal (${subtotal:,.2f}) + Tax (${tax:,.2f}) matches grand total (${total:,.2f}).",
            })
        else:
            results.append({
                "check": "Subtotal + Tax == Total",
                "status": "FAIL",
                "message": (
                    f"Total mismatch: Subtotal (${subtotal:,.2f}) + Tax (${tax:,.2f}) = "
                    f"${expected_total:,.2f}, but stated Total is ${total:,.2f} (Discrepancy: ${total_diff:+,.2f})."
                ),
            })
    else:
        results.append({
            "check": "Subtotal + Tax == Total",
            "status": "FAIL",
            "message": "Cannot reconcile grand total: Subtotal or Grand Total is missing.",
        })

    return results


def is_all_passed(validation_results: List[Dict[str, str]]) -> bool:
    """Returns True if every validation check has status 'PASS'."""
    return all(r.get("status") == "PASS" for r in validation_results)
