"""
generate_samples.py - Synthetic invoice generator using Pillow and NumPy.

Generates 3 realistic synthetic invoices in PNG format:
1. sample_1_valid.png: Clean, perfectly calculated invoice.
2. sample_2_mismatch.png: Clean invoice with an intentional grand total mismatch.
3. sample_3_noisy_rotated.png: Invoice with slight rotation and realistic scan noise.

Each invoice contains:
- Vendor name, address, contact details
- Invoice number, issue date, due date
- Customer "Bill To" section
- Line-item table (description, quantity, unit price, amount)
- Financial breakdown (subtotal, tax, grand total)
- Payment notes and footer
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a truetype font with cross-platform fallback."""
    candidates = (
        ["arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf", "segoeuib.ttf"]
        if bold
        else ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "segoeui.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def get_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    """Calculate the rendered width of text in pixels."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def render_invoice_canvas(
    data: Dict[str, Any],
    theme_color: Tuple[int, int, int] = (30, 58, 138),
    accent_bg: Tuple[int, int, int] = (241, 245, 249),
) -> Image.Image:
    """
    Renders a realistic, publication-quality invoice image using Pillow.

    Args:
        data: Invoice metadata, line items, and totals.
        theme_color: RGB tuple for primary header / table banner color.
        accent_bg: RGB tuple for light highlights and zebra rows.

    Returns:
        Pillow Image object.
    """
    width, height = 1000, 1320
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_brand = load_font(28, bold=True)
    font_title = load_font(32, bold=True)
    font_header = load_font(14, bold=True)
    font_bold = load_font(14, bold=True)
    font_body = load_font(14, bold=False)
    font_small = load_font(11, bold=False)
    font_total = load_font(20, bold=True)

    # Margins and bounds
    margin_x = 65
    content_width = width - (2 * margin_x)
    right_margin = width - margin_x

    # 1. Top Decorative Brand Bar
    draw.rectangle([(margin_x, 35), (right_margin, 40)], fill=theme_color)

    # 2. Header Section: Vendor Info (Left) & INVOICE Title (Right)
    # Vendor Block
    draw.text((margin_x, 55), data["vendor_name"], fill=(15, 23, 42), font=font_brand)
    draw.text((margin_x, 95), data["vendor_address"], fill=(71, 85, 105), font=font_body)
    draw.text((margin_x, 115), data["vendor_contact"], fill=(71, 85, 105), font=font_body)

    # Title & Metadata Block (Right aligned)
    inv_title = "INVOICE"
    t_w = get_text_width(draw, inv_title, font_title)
    draw.text((right_margin - t_w, 52), inv_title, fill=theme_color, font=font_title)

    meta_x_label = right_margin - 210
    meta_x_val = right_margin

    def draw_meta_row(y_pos: int, label: str, value: str):
        draw.text((meta_x_label, y_pos), label, fill=(100, 116, 139), font=font_bold)
        val_w = get_text_width(draw, value, font_bold)
        draw.text((meta_x_val - val_w, y_pos), value, fill=(15, 23, 42), font=font_bold)

    draw_meta_row(98, "Invoice Number:", data["invoice_number"])
    draw_meta_row(120, "Issue Date:", data["date"])
    draw_meta_row(142, "Payment Due:", data.get("due_date", "Upon Receipt"))

    # Divider
    draw.line([(margin_x, 175), (right_margin, 175)], fill=(226, 232, 240), width=1)

    # 3. Bill-To & Ship-To Section
    billto_y = 195
    # Card background for Bill To
    draw.rounded_rectangle(
        [(margin_x, billto_y), (margin_x + 380, billto_y + 105)],
        radius=6,
        fill=accent_bg,
        outline=(226, 232, 240),
    )
    draw.text((margin_x + 18, billto_y + 12), "BILLED TO:", fill=theme_color, font=font_header)
    draw.text((margin_x + 18, billto_y + 36), data["client_name"], fill=(15, 23, 42), font=font_bold)
    draw.text((margin_x + 18, billto_y + 58), data["client_address"], fill=(71, 85, 105), font=font_body)
    draw.text((margin_x + 18, billto_y + 78), data["client_contact"], fill=(71, 85, 105), font=font_body)

    # Status badge / Payment Info Card
    draw.rounded_rectangle(
        [(right_margin - 380, billto_y), (right_margin, billto_y + 105)],
        radius=6,
        fill=accent_bg,
        outline=(226, 232, 240),
    )
    draw.text((right_margin - 362, billto_y + 12), "PAYMENT DETAILS:", fill=theme_color, font=font_header)
    draw.text((right_margin - 362, billto_y + 36), f"Payment Method: {data.get('payment_method', 'Bank Transfer (ACH)')}", fill=(71, 85, 105), font=font_body)
    draw.text((right_margin - 362, billto_y + 58), f"Currency: {data.get('currency', 'USD ($)')}", fill=(71, 85, 105), font=font_body)
    draw.text((right_margin - 362, billto_y + 78), f"Account Reference: {data['invoice_number']}", fill=(71, 85, 105), font=font_body)

    # 4. Line Items Table
    table_top = 330
    header_height = 36
    draw.rounded_rectangle(
        [(margin_x, table_top), (right_margin, table_top + header_height)],
        radius=4,
        fill=theme_color,
    )

    # Column coordinate boundaries
    col_desc_x = margin_x + 16
    col_qty_right = margin_x + 520
    col_price_right = margin_x + 690
    col_amount_right = right_margin - 16

    # Table Header text
    draw.text((col_desc_x, table_top + 10), "DESCRIPTION", fill=(255, 255, 255), font=font_header)
    q_hdr_w = get_text_width(draw, "QTY", font_header)
    draw.text((col_qty_right - q_hdr_w, table_top + 10), "QTY", fill=(255, 255, 255), font=font_header)
    p_hdr_w = get_text_width(draw, "UNIT PRICE", font_header)
    draw.text((col_price_right - p_hdr_w, table_top + 10), "UNIT PRICE", fill=(255, 255, 255), font=font_header)
    a_hdr_w = get_text_width(draw, "AMOUNT", font_header)
    draw.text((col_amount_right - a_hdr_w, table_top + 10), "AMOUNT", fill=(255, 255, 255), font=font_header)

    # Table Rows
    curr_y = table_top + header_height
    row_height = 46

    for idx, item in enumerate(data["line_items"]):
        row_bg = accent_bg if (idx % 2 == 1) else (255, 255, 255)
        draw.rectangle([(margin_x, curr_y), (right_margin, curr_y + row_height)], fill=row_bg)

        # Row border line
        draw.line([(margin_x, curr_y + row_height), (right_margin, curr_y + row_height)], fill=(226, 232, 240), width=1)

        # Content
        text_y = curr_y + 14
        draw.text((col_desc_x, text_y), item["description"], fill=(30, 41, 59), font=font_body)

        qty_str = str(item["qty"])
        qw = get_text_width(draw, qty_str, font_bold)
        draw.text((col_qty_right - qw, text_y), qty_str, fill=(30, 41, 59), font=font_bold)

        price_str = f"${item['unit_price']:,.2f}"
        pw = get_text_width(draw, price_str, font_bold)
        draw.text((col_price_right - pw, text_y), price_str, fill=(30, 41, 59), font=font_bold)

        amt_str = f"${item['amount']:,.2f}"
        aw = get_text_width(draw, amt_str, font_bold)
        draw.text((col_amount_right - aw, text_y), amt_str, fill=(15, 23, 42), font=font_bold)

        curr_y += row_height

    # 5. Financial Summary / Totals Box (Right Aligned)
    summary_top = curr_y + 25
    summary_width = 380
    summary_x1 = right_margin - summary_width
    summary_x2 = right_margin

    # Background frame for totals
    draw.rounded_rectangle(
        [(summary_x1, summary_top), (summary_x2, summary_top + 165)],
        radius=6,
        fill=accent_bg,
        outline=(203, 213, 225),
    )

    def draw_total_row(y: int, label: str, val: float, is_grand_total: bool = False):
        font_to_use = font_total if is_grand_total else font_bold
        val_str = f"${val:,.2f}"
        color = theme_color if is_grand_total else (30, 41, 59)

        if is_grand_total:
            # Subtle accent fill for total row
            draw.rectangle([(summary_x1 + 1, y - 8), (summary_x2 - 1, y + 36)], fill=(226, 232, 240))
            draw.line([(summary_x1, y - 8), (summary_x2, y - 8)], fill=theme_color, width=2)

        draw.text((summary_x1 + 20, y + 4), label, fill=color, font=font_to_use)
        vw = get_text_width(draw, val_str, font_to_use)
        draw.text((summary_x2 - 20 - vw, y + 4), val_str, fill=color, font=font_to_use)

    draw_total_row(summary_top + 12, "Subtotal", data["subtotal"])
    tax_label = f"Tax ({data.get('tax_rate_pct', 10):.0f}%)"
    draw_total_row(summary_top + 52, tax_label, data["tax"])
    draw_total_row(summary_top + 104, "Total Amount Due", data["total"], is_grand_total=True)

    # 6. Terms, Notes & Instructions (Below Totals Box)
    notes_top = summary_top + 185
    draw.text((margin_x, notes_top), "TERMS & CONDITIONS:", fill=theme_color, font=font_header)
    draw.text(
        (margin_x, notes_top + 24),
        "1. Please include invoice number on your payment voucher or wire reference.",
        fill=(100, 116, 139),
        font=font_small,
    )
    draw.text(
        (margin_x, notes_top + 42),
        "2. Payments received after the due date may be subject to a 1.5% monthly late fee.",
        fill=(100, 116, 139),
        font=font_small,
    )
    draw.text(
        (margin_x, notes_top + 60),
        "3. For billing questions or discrepancies, please contact billing@support.com.",
        fill=(100, 116, 139),
        font=font_small,
    )

    # 7. Page Footer
    footer_y = height - 55
    draw.line([(margin_x, footer_y), (right_margin, footer_y)], fill=(226, 232, 240), width=1)
    footer_text = f"Thank you for choosing {data['vendor_name']}! | Authorized Commercial Invoice"
    fw = get_text_width(draw, footer_text, font_small)
    draw.text((int((width - fw) / 2), footer_y + 15), footer_text, fill=(148, 163, 184), font=font_small)

    return img


def apply_scanner_distortion_and_noise(img: Image.Image, angle: float = -2.6) -> Image.Image:
    """
    Simulates a scanned physical paper document:
    - Rotates image slightly
    - Adds Gaussian noise / sensor noise
    - Injects subtle scanner speckles and faint paper grain
    """
    # 1. Rotate with white fill
    rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # 2. Convert to NumPy array for realistic noise injection
    arr = np.array(rotated, dtype=np.float32)

    # Gaussian scanner noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=9.0, size=arr.shape)
    noisy = arr + noise

    # Subtle horizontal scanning sensor artifacts / faint banding
    banding = np.sin(np.linspace(0, 30 * np.pi, noisy.shape[0]))[:, None, None] * 3.5
    noisy += banding

    # Scanner glass dust/speckles
    speckle_count = 260
    for _ in range(speckle_count):
        sx = rng.integers(20, noisy.shape[1] - 20)
        sy = rng.integers(20, noisy.shape[0] - 20)
        speckle_val = rng.integers(40, 130)
        noisy[sy : sy + 2, sx : sx + 2] = speckle_val

    # Clip and convert back to PIL Image
    noisy_uint8 = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_uint8)


def generate_all_samples(output_dir: Path) -> List[Path]:
    """
    Generates all 3 sample invoices and saves them as PNG files.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # -------------------------------------------------------------
    # Sample 1: Completely Correct Invoice
    # -------------------------------------------------------------
    sample_1_data = {
        "vendor_name": "Apex Cloud Technologies Inc.",
        "vendor_address": "100 Innovation Way, Suite 400, San Jose, CA 95134",
        "vendor_contact": "billing@apexcloud.io | +1 (555) 019-2834",
        "client_name": "Acme Digital Services LLC",
        "client_address": "742 Evergreen Terrace, Springfield, IL 62704",
        "client_contact": "Attn: Accounts Payable | finance@acmedigital.com",
        "invoice_number": "INV-2026-1042",
        "date": "2026-10-02",
        "due_date": "2026-11-01",
        "payment_method": "ACH / Electronic Wire",
        "currency": "USD ($)",
        "tax_rate_pct": 10.0,
        "line_items": [
            {"description": "Cloud Infrastructure Dedicated Hosting (Monthly)", "qty": 1, "unit_price": 450.00, "amount": 450.00},
            {"description": "Database High-Availability Optimization", "qty": 4, "unit_price": 125.00, "amount": 500.00},
            {"description": "Enterprise SSL Security Certificate Renewal", "qty": 2, "unit_price": 75.00, "amount": 150.00},
        ],
        "subtotal": 1100.00,  # 450 + 500 + 150 = 1100.00
        "tax": 110.00,        # 10% of 1100 = 110.00
        "total": 1210.00,      # 1100 + 110 = 1210.00 (VERIFIED CORRECT)
    }

    img1 = render_invoice_canvas(
        sample_1_data,
        theme_color=(30, 58, 138),    # Deep Slate Blue
        accent_bg=(241, 245, 249),
    )
    path1 = output_dir / "sample_1_valid.png"
    img1.save(path1, "PNG")
    generated_files.append(path1)

    # -------------------------------------------------------------
    # Sample 2: Deliberate Total Mismatch
    # -------------------------------------------------------------
    # Line items: 3*220=660, 1*340=340, 5*40=200 -> Sum = 1200.00
    # Tax: 10% = 120.00
    # True Total: 1320.00
    # Stated Total on Invoice: 1450.00 (DELIBERATE MISMATCH of $130)
    sample_2_data = {
        "vendor_name": "Nova Logistics & Supply Co.",
        "vendor_address": "450 Industrial Parkway, Dock 12, Chicago, IL 60607",
        "vendor_contact": "accounts@novalogistics.com | +1 (555) 438-9102",
        "client_name": "Global Freight Partners Inc.",
        "client_address": "88 Harbor Blvd, Suite 210, Seattle, WA 98101",
        "client_contact": "Attn: Procurement Team | invoices@globalfreight.com",
        "invoice_number": "INV-2026-2189",
        "date": "2026-10-08",
        "due_date": "2026-11-07",
        "payment_method": "Corporate Card / Wire",
        "currency": "USD ($)",
        "tax_rate_pct": 10.0,
        "line_items": [
            {"description": "Pallet Freight Shipping - Zone 3 Distribution", "qty": 3, "unit_price": 220.00, "amount": 660.00},
            {"description": "Climate-Controlled Warehouse Storage (Monthly)", "qty": 1, "unit_price": 340.00, "amount": 340.00},
            {"description": "Standard Heavy-Duty Packaging Materials", "qty": 5, "unit_price": 40.00, "amount": 200.00},
        ],
        "subtotal": 1200.00,  # 660 + 340 + 200 = 1200.00
        "tax": 120.00,        # 10% of 1200 = 120.00
        "total": 1450.00,      # DELIBERATE MISMATCH: Should be 1320.00, stated as 1450.00
    }

    img2 = render_invoice_canvas(
        sample_2_data,
        theme_color=(15, 118, 110),   # Teal / Emerald
        accent_bg=(240, 253, 250),
    )
    path2 = output_dir / "sample_2_mismatch.png"
    img2.save(path2, "PNG")
    generated_files.append(path2)

    # -------------------------------------------------------------
    # Sample 3: Slightly Rotated with Realistic Scanner Noise
    # -------------------------------------------------------------
    sample_3_data = {
        "vendor_name": "Crestview Office Solutions",
        "vendor_address": "820 Commerce Blvd, Suite 500, Austin, TX 78701",
        "vendor_contact": "invoicing@crestviewoffice.com | +1 (555) 782-4411",
        "client_name": "Pinnacle Consulting Group",
        "client_address": "1200 Market Street, 14th Floor, Denver, CO 80202",
        "client_contact": "Attn: Office Operations | finance@pinnacle.org",
        "invoice_number": "INV-2026-3751",
        "date": "2026-10-12",
        "due_date": "2026-11-11",
        "payment_method": "Commercial Check / ACH",
        "currency": "USD ($)",
        "tax_rate_pct": 8.0,
        "line_items": [
            {"description": "Ergonomic Lumbar Mesh Desk Chair (Black)", "qty": 2, "unit_price": 280.00, "amount": 560.00},
            {"description": "Dual Heavy-Duty Gas Spring Monitor Arm", "qty": 3, "unit_price": 85.00, "amount": 255.00},
            {"description": "Thunderbolt / USB-C Multiport Docking Station", "qty": 2, "unit_price": 110.00, "amount": 220.00},
        ],
        "subtotal": 1035.00,  # 560 + 255 + 220 = 1035.00
        "tax": 82.80,         # 8% of 1035 = 82.80
        "total": 1117.80,     # 1035 + 82.80 = 1117.80 (Math correct, visual distortion tested)
    }

    img3_clean = render_invoice_canvas(
        sample_3_data,
        theme_color=(51, 65, 85),     # Charcoal / Slate
        accent_bg=(248, 250, 252),
    )
    img3_distorted = apply_scanner_distortion_and_noise(img3_clean, angle=-2.6)
    path3 = output_dir / "sample_3_noisy_rotated.png"
    img3_distorted.save(path3, "PNG")
    generated_files.append(path3)

    return generated_files


def main():
    # Resolve samples directory relative to script
    samples_dir = Path(__file__).resolve().parent / "samples"
    print(f"Generating synthetic invoices into: {samples_dir.resolve()} ...")
    files = generate_all_samples(samples_dir)
    print("\nSuccessfully generated samples:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f" - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
