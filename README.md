# 🧾 Invoice Extractor & Validator

An end-to-end Python pipeline and interactive Streamlit web application that extracts structured data from commercial invoices (PNG, JPG, PDF), applies document image preprocessing (deskewing, binarization, denoising), parses tabular line items, and reconciles arithmetic totals.

---

## 📌 Features

- **PyMuPDF PDF-to-Image**: Converts the first page of multi-page or single-page PDF invoices directly into high-resolution images in pure Python—no Poppler or external C++ tools needed.
- **OpenCV Image Preprocessing**:
  - Grayscale conversion.
  - Median filter noise reduction to clean scanner speckles and sensor noise.
  - Adaptive Gaussian thresholding for binarization.
  - Automatic deskewing via `cv2.minAreaRect` on connected text pixels.
- **Tesseract OCR (PSM 6)**: Extracts text assuming a uniform block layout.
- **Robust Regex & Slip Correction**:
  - Handles common OCR letter-to-digit confusions: `O`/`0`, `S`/`5`, `Z`/`2`, `a`/`4`, `b`/`8`.
  - Reconciles space-separated cents (`1,100 00` -> `1,100.00`), double currency symbols (`$$60` -> `$560`), and malformed decimal points.
  - Extracts Vendor Name, Invoice Number, Issue Date, Line Items (`description`, `qty`, `unit_price`, `amount`), Subtotal, Tax, and Grand Total.
- **Automated Validation**:
  - Checks presence of all critical header fields.
  - Reconciles line-item sum: $\sum(\text{Amount}) = \text{Subtotal}$ (tolerance $\pm0.01$).
  - Reconciles grand total: $\text{Subtotal} + \text{Tax} = \text{Grand Total}$ (tolerance $\pm0.01$).
  - Correctly flags arithmetic discrepancies (tested against deliberate mismatches).
- **Interactive Streamlit Web Interface**:
  - Side-by-side comparison of Original vs Preprocessed image.
  - Editable inputs for all header fields.
  - Interactive tabular line-items editor via `st.data_editor`.
  - Instant reactive re-validation as users edit values.
  - One-click CSV and JSON exports.

---

## 📁 Project Directory Structure

```text
invoice-extractor/
├── app.py                # Streamlit web application
├── preprocess.py         # PDF conversion, image preprocessing (deskew/threshold), and OCR
├── extractor.py          # Regex extraction rules and OCR slip correction
├── validator.py          # Arithmetic & business logic validation engine
├── generate_samples.py   # Synthetic invoice generator (Pillow & NumPy)
├── test_pipeline.py      # Automated CLI test suite for all samples
├── requirements.txt      # Python package dependencies
├── samples/              # Synthetic test invoices
│   ├── sample_1_valid.png          # Valid, fully reconciled invoice
│   ├── sample_2_mismatch.png       # Invoice with deliberate total mismatch
│   └── sample_3_noisy_rotated.png  # Invoice with rotation & scanner noise
└── README.md             # Project documentation
```

---

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR Engine

Tesseract OCR must be installed on your operating system:

#### **Windows**:
1. Download the official Windows installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to default location: `C:\Program Files\Tesseract-OCR`.
3. The project automatically detects `tesseract.exe` at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   *(Optional)* Add `C:\Program Files\Tesseract-OCR` to your system `PATH`.

#### **Linux (Ubuntu / Debian)**:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

#### **macOS (Homebrew)**:
```bash
brew install tesseract
```

---

## 🚀 Running the Project

### 1. Generate Synthetic Sample Invoices
Generate the 3 test PNG invoices into `samples/`:
```bash
python generate_samples.py
```

### 2. Run Preprocessing CLI Test
Inspect the OCR output from all sample files:
```bash
python preprocess.py
```

### 3. Run Pipeline Test & Verification Table
Verify the full extraction and validation across all 3 samples:
```bash
python test_pipeline.py
```

### 4. Launch the Interactive Streamlit Web UI
```bash
streamlit run app.py
```
Or specify a custom port:
```bash
streamlit run app.py --server.port 8501
```

---

## 🧪 Pipeline Test Results

Running `test_pipeline.py` executes the entire pipeline across all three samples:

| Sample File | Detected Angle | Extracted Invoice # | Stated Subtotal | Stated Tax | Stated Total | Line Items Sum | Validation Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sample_1_valid.png` | $0.00^\circ$ | `INV-2026-1042` | \$1,100.00 | \$110.00 | \$1,210.00 | \$1,100.00 | **PASSED** |
| `sample_2_mismatch.png` | $0.00^\circ$ | `INV-2026-2189` | \$1,200.00 | \$120.00 | \$1,450.00 | \$1,200.00 | **FLAGGED (MISMATCH)** |
| `sample_3_noisy_rotated.png` | $2.59^\circ$ | `INV-2026-3751` | \$1,035.00 | \$82.80 | \$1,117.80 | \$1,035.00 | **PASSED** |

- **Sample 1**: Fully passes all arithmetic and field checks.
- **Sample 2**: Line items sum matches subtotal (\$1,200), but stated total (\$1,450) differs from expected total (\$1,320), correctly triggering an alert.
- **Sample 3**: Automatically deskewed by $2.59^\circ$, filtered for noise, and fully reconciled.

---

## ⚠️ Limitations & Considerations

While this project handles common invoice formats and scanner artifacts, keep the following considerations in mind:

1. **Rule-Based & Layout-Dependent**:
   - Field extraction uses regular expressions tailored to standard commercial invoices (e.g., labels like `Invoice Number`, `Date`, `Subtotal`, `Total`).
   - If an invoice omits standard labels, places values in unconventional visual quadrants, or splits values across disparate columns, rule-based matching may need tuning.
2. **Multi-Column Complexity**:
   - Tesseract `--psm 6` assumes a single uniform block of text. For invoices with complex nested tables or parallel independent columns, regional OCR zoning (bounding box segmentation) or LayoutLM models provide greater flexibility.
3. **Severe Optical Distortion**:
   - Preprocessing corrects planar skew up to $\pm45^\circ$. Nonlinear distortions (e.g. crinkled paper, curved page folds, or severe perspective distortion) require camera perspective transforms (`cv2.warpPerspective`).
4. **Resolution Requirements**:
   - Low-resolution scans (< 150 DPI) can introduce OCR letter degradation. High-DPI scans (300 DPI) produce optimal OCR accuracy.
