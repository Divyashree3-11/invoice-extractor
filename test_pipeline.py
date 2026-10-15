"""
test_pipeline.py - Comprehensive CLI test for preprocess -> extractor -> validator.

Runs on all 3 sample invoices and displays a structured results table.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from preprocess import process_document
from extractor import extract_invoice_data
from validator import validate_invoice, is_all_passed


def main():
    samples_dir = Path(__file__).resolve().parent / "samples"
    sample_files = sorted(list(samples_dir.glob("*.png")))

    print("\n" + "=" * 90)
    print(" INVOICE EXTRACTOR & VALIDATOR PIPELINE TEST ")
    print("=" * 90 + "\n")

    overall_summary = []

    for path in sample_files:
        print(f"\nProcessing: {path.name}")
        print("-" * 90)

        # 1. Preprocess & OCR
        orig, proc, skew_angle, ocr_text, pdf_native_text = process_document(path)
        used_native = pdf_native_text is not None and len(pdf_native_text.strip()) >= 50
        print(f" * Preprocessing: Deskew detected {skew_angle:.2f} deg rotation. {'(PDF native text used, OCR skipped)' if used_native else '(OCR applied)'}")

        # 2. Extract
        data = extract_invoice_data(ocr_text, pdf_native_text=pdf_native_text)
        print(f" * Extracted Vendor:    {data['vendor']}")
        print(f" * Extracted Invoice #: {data['invoice_number']}")
        print(f" * Extracted Date:      {data['date']}")
        print(f" * Financial Summary:   Subtotal: ${data['subtotal'] or 0:,.2f} | Tax: ${data['tax'] or 0:,.2f} | Total: ${data['total'] or 0:,.2f}")
        print(f" * Line Items Found:    {len(data['line_items'])} item(s)")
        for it in data["line_items"]:
            print(f"     - {it['description'][:35]:<35} | Qty: {it['qty']} | Unit: ${it['unit_price']:,.2f} | Amt: ${it['amount']:,.2f}")

        # 3. Validate
        checks = validate_invoice(data)
        all_ok = is_all_passed(checks)

        print("\n * Validation Results:")
        for c in checks:
            badge = "[PASS]" if c["status"] == "PASS" else "[FAIL]"
            print(f"     {badge} {c['check']}: {c['message']}")

        overall_status = "PASSED" if all_ok else "FLAGGED (MISMATCH)"
        overall_summary.append({
            "File": path.name,
            "Vendor": data["vendor"] or "Unknown",
            "Invoice #": data["invoice_number"] or "N/A",
            "Subtotal": f"${data['subtotal'] or 0:,.2f}",
            "Tax": f"${data['tax'] or 0:,.2f}",
            "Total": f"${data['total'] or 0:,.2f}",
            "Items Sum": f"${sum(it.get('amount', 0) for it in data['line_items']):,.2f}",
            "Status": overall_status,
        })

    # Summary Table
    print("\n" + "=" * 90)
    print(" PIPELINE TEST RESULTS SUMMARY TABLE")
    print("=" * 90)
    headers = ["File", "Invoice #", "Subtotal", "Tax", "Total", "Items Sum", "Status"]
    header_line = f"{headers[0]:<27} | {headers[1]:<14} | {headers[2]:<10} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]}"
    print(header_line)
    print("-" * len(header_line))
    for row in overall_summary:
        print(f"{row['File']:<27} | {row['Invoice #']:<14} | {row['Subtotal']:<10} | {row['Tax']:<8} | {row['Total']:<10} | {row['Items Sum']:<10} | {row['Status']}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
