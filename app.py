"""
app.py - Streamlit web interface for Invoice Extractor & Validator.

Features:
- File uploader (PNG, JPG, PDF)
- Quick sample invoice selector
- Side-by-side view: Original vs Preprocessed image
- Editable header fields (Vendor, Invoice #, Date, Subtotal, Tax, Total)
- Interactive line items table with st.data_editor
- Reactive validation results (re-runs when user edits values)
- Download buttons for CSV (line items) and JSON (complete invoice data)
- Spinner and graceful error handling
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import streamlit as st

from preprocess import process_document, load_document, preprocess_image, run_ocr
from extractor import extract_invoice_data
from validator import validate_invoice, is_all_passed


def main():
    st.set_page_config(
        page_title="Invoice Extractor & Validator",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧾 Intelligent Invoice Extractor & Validator")
    st.markdown(
        "Upload an invoice image (PNG, JPG) or PDF to preprocess, extract structured data with OCR, "
        "and automatically validate financial calculations and totals."
    )

    # -------------------------------------------------------------
    # Sidebar: File Upload & Quick Samples
    # -------------------------------------------------------------
    st.sidebar.header("📁 Document Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Invoice (PNG, JPG, PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload an invoice image or PDF document.",
    )

    samples_dir = Path(__file__).resolve().parent / "samples"
    sample_files = sorted(list(samples_dir.glob("*.png"))) if samples_dir.exists() else []

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Or Select a Sample Invoice")
    
    selected_sample_path = None
    for sample in sample_files:
        label = sample.stem.replace("_", " ").title()
        if st.sidebar.button(label, key=f"btn_{sample.name}", use_container_width=True):
            st.session_state["active_file_path"] = str(sample)
            # Clear previous processing cache when a new sample is clicked
            st.session_state.pop("processed_data", None)

    active_file_path = st.session_state.get("active_file_path")

    # Determine input source
    source = None
    display_name = ""
    if uploaded_file is not None:
        source = uploaded_file.read()
        display_name = uploaded_file.name
        # If user uploaded a new file, invalidate cached sample
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state["last_uploaded"] = uploaded_file.name
            st.session_state.pop("processed_data", None)
    elif active_file_path and Path(active_file_path).exists():
        source = Path(active_file_path)
        display_name = source.name

    if source is None:
        st.info("👈 Upload an invoice file from the sidebar or click one of the sample buttons to get started.")
        
        # Display sample cards
        if sample_files:
            st.subheader("Available Test Samples:")
            cols = st.columns(len(sample_files))
            for i, p in enumerate(sample_files):
                with cols[i]:
                    st.image(str(p), caption=p.stem, use_container_width=True)
        return

    # -------------------------------------------------------------
    # Document Processing & OCR
    # -------------------------------------------------------------
    if "processed_data" not in st.session_state:
        with st.spinner(f"Processing '{display_name}' (Deskewing, Thresholding, Tesseract OCR)..."):
            try:
                orig, proc, skew_angle, ocr_text, pdf_native_text = process_document(source)
                extracted = extract_invoice_data(ocr_text, pdf_native_text=pdf_native_text)

                st.session_state["processed_data"] = {
                    "original": orig,
                    "processed": proc,
                    "skew_angle": skew_angle,
                    "ocr_text": ocr_text,
                    "pdf_native_text": pdf_native_text,
                    "extracted": extracted,
                }
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")
                return

    proc_state = st.session_state["processed_data"]
    orig_img = proc_state["original"]
    proc_img = proc_state["processed"]
    skew_angle = proc_state["skew_angle"]
    extracted = proc_state["extracted"]

    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 **Current:** `{display_name}`\n\n📐 **Skew Angle:** `{skew_angle:.2f}°`")
    if st.sidebar.button("🔄 Re-process Document", use_container_width=True):
        st.session_state.pop("processed_data", None)
        st.rerun()

    # -------------------------------------------------------------
    # Visual Comparison: Original vs Preprocessed
    # -------------------------------------------------------------
    st.subheader("🖼️ Image Preprocessing (Deskewed & Binarized)")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.caption("📷 Original Document")
        st.image(orig_img, use_container_width=True)
    with col_img2:
        st.caption(f"⚙️ Preprocessed Binary (Deskew: {skew_angle:.2f}°)")
        st.image(proc_img, use_container_width=True)

    # -------------------------------------------------------------
    # Extracted Header Fields (Editable Inputs)
    # -------------------------------------------------------------
    st.subheader("📝 Extracted Invoice Header & Totals (Editable)")
    st.caption("You can modify extracted fields below; arithmetic checks will re-validate reactively.")

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        vendor_val = st.text_input("🏢 Vendor Name", value=extracted.get("vendor") or "")
    with h_col2:
        inv_num_val = st.text_input("🔢 Invoice Number", value=extracted.get("invoice_number") or "")
    with h_col3:
        date_val = st.text_input("📅 Invoice Date", value=extracted.get("date") or "")

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        subtotal_val = st.number_input(
            "💵 Stated Subtotal ($)",
            value=float(extracted.get("subtotal") or 0.0),
            step=10.0,
            format="%.2f",
        )
    with t_col2:
        tax_val = st.number_input(
            "🏛️ Stated Tax ($)",
            value=float(extracted.get("tax") or 0.0),
            step=5.0,
            format="%.2f",
        )
    with t_col3:
        total_val = st.number_input(
            "💰 Stated Grand Total ($)",
            value=float(extracted.get("total") or 0.0),
            step=10.0,
            format="%.2f",
        )

    # -------------------------------------------------------------
    # Line Items Table (Interactive st.data_editor)
    # -------------------------------------------------------------
    st.subheader("📋 Line Items")
    st.caption("Add, delete, or edit item quantities and unit prices directly in the table.")

    raw_items = extracted.get("line_items") or []
    if raw_items:
        df_items = pd.DataFrame(raw_items)
    else:
        df_items = pd.DataFrame(columns=["description", "qty", "unit_price", "amount"])

    # Ensure correct data types
    if "qty" in df_items.columns:
        df_items["qty"] = pd.to_numeric(df_items["qty"], errors="coerce").fillna(1).astype(int)
    if "unit_price" in df_items.columns:
        df_items["unit_price"] = pd.to_numeric(df_items["unit_price"], errors="coerce").fillna(0.0).astype(float)
    if "amount" in df_items.columns:
        df_items["amount"] = pd.to_numeric(df_items["amount"], errors="coerce").fillna(0.0).astype(float)

    edited_items_df = st.data_editor(
        df_items,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "description": st.column_config.TextColumn("Description", required=True),
            "qty": st.column_config.NumberColumn("Quantity", min_value=1, step=1, format="%d"),
            "unit_price": st.column_config.NumberColumn("Unit Price ($)", min_value=0.0, step=1.0, format="$%.2f"),
            "amount": st.column_config.NumberColumn("Amount ($)", min_value=0.0, step=1.0, format="$%.2f"),
        },
        key="line_items_data_editor",
    )

    # Convert edited DataFrame back to list of dicts for validation
    active_line_items = edited_items_df.to_dict(orient="records")

    # -------------------------------------------------------------
    # Reactive Validation
    # -------------------------------------------------------------
    st.subheader("✅ Validation & Arithmetic Verification")

    active_invoice_data = {
        "vendor": vendor_val.strip() if vendor_val else None,
        "invoice_number": inv_num_val.strip() if inv_num_val else None,
        "date": date_val.strip() if date_val else None,
        "subtotal": float(subtotal_val) if subtotal_val > 0 else None,
        "tax": float(tax_val),
        "total": float(total_val) if total_val > 0 else None,
        "line_items": active_line_items,
    }

    validation_checks = validate_invoice(active_invoice_data, tolerance=0.01)
    all_passed = is_all_passed(validation_checks)

    if all_passed:
        st.success("🎉 **STATUS: ALL CHECKS PASSED** — Invoice arithmetic and critical fields are fully reconciled.")
    else:
        st.error("⚠️ **STATUS: VALIDATION FLAGGED** — Discrepancies or missing fields detected. Review details below.")

    for c in validation_checks:
        if c["status"] == "PASS":
            st.success(f"**{c['check']}**: {c['message']}")
        else:
            st.error(f"**{c['check']}**: {c['message']}")

    # -------------------------------------------------------------
    # Export & Downloads (CSV & JSON)
    # -------------------------------------------------------------
    st.subheader("💾 Export Extracted Data")
    d_col1, d_col2 = st.columns(2)

    # 1. CSV Download
    csv_data = edited_items_df.to_csv(index=False).encode("utf-8")
    with d_col1:
        st.download_button(
            label="📥 Download Line Items (CSV)",
            data=csv_data,
            file_name=f"{active_invoice_data['invoice_number'] or 'invoice'}_line_items.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 2. JSON Download
    full_export = {
        "metadata": {
            "vendor": active_invoice_data["vendor"],
            "invoice_number": active_invoice_data["invoice_number"],
            "date": active_invoice_data["date"],
        },
        "financials": {
            "subtotal": active_invoice_data["subtotal"],
            "tax": active_invoice_data["tax"],
            "total": active_invoice_data["total"],
            "calculated_items_sum": round(sum((it.get("amount") or 0.0) for it in active_line_items), 2),
        },
        "line_items": active_line_items,
        "validation": {
            "is_valid": all_passed,
            "checks": validation_checks,
        },
    }
    json_data = json.dumps(full_export, indent=2).encode("utf-8")
    with d_col2:
        st.download_button(
            label="📥 Download Complete Invoice (JSON)",
            data=json_data,
            file_name=f"{active_invoice_data['invoice_number'] or 'invoice'}_extracted.json",
            mime="application/json",
            use_container_width=True,
        )

    # Raw text expander
    with st.expander("🔍 View Raw Extracted Text"):
        pdf_native = proc_state.get("pdf_native_text")
        if pdf_native:
            st.caption("📄 Source: PyMuPDF native text layer (PDF fast-path — no OCR used)")
            st.text(pdf_native)
        else:
            st.caption("📸 Source: Tesseract OCR on preprocessed image")
            st.text(proc_state.get("ocr_text", ""))


if __name__ == "__main__":
    main()
