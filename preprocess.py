"""
preprocess.py - Document loading, image preprocessing, and OCR pipeline.

Pipeline:
1. Load image (PNG/JPG) or first page of PDF via PyMuPDF.
2. Convert to Grayscale.
3. Denoise using median blur / fastNlMeans.
4. Adaptive Thresholding (Gaussian adaptive binarization).
5. Deskew using cv2.minAreaRect on text pixels.
6. OCR execution using pytesseract (image_to_string with PSM 6).
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image
import pymupdf
import pytesseract

# Configure Tesseract binary path if not in standard PATH (Windows support)
if not shutil.which("tesseract"):
    for t_path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]:
        if os.path.exists(t_path):
            pytesseract.pytesseract.tesseract_cmd = t_path
            break


def load_document(source: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Load an image or the first page of a PDF and return as an RGB NumPy array.

    Args:
        source: File path, bytes, PIL Image, or NumPy array.

    Returns:
        RGB image as np.ndarray (H, W, 3) uint8.
    """
    if isinstance(source, np.ndarray):
        if len(source.shape) == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        elif source.shape[2] == 3:
            return source.copy()
        return source

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))

    # If bytes or file path
    is_pdf = False
    file_bytes: Optional[bytes] = None

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.suffix.lower() == ".pdf":
            is_pdf = True
        else:
            img = cv2.imread(str(source_path))
            if img is None:
                raise ValueError(f"Could not load image from path: {source}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(source, bytes):
        file_bytes = source
        if file_bytes.startswith(b"%PDF"):
            is_pdf = True

    if is_pdf:
        # Load PDF first page via PyMuPDF (no poppler required)
        if file_bytes is not None:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        else:
            doc = pymupdf.open(str(source))

        if len(doc) == 0:
            raise ValueError("PDF document contains no pages.")

        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        # Convert pixmap to numpy array
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        doc.close()
        return img_np

    # Bytes for non-PDF image
    if file_bytes is not None:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided bytes.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise TypeError(f"Unsupported document source type: {type(source)}")


def deskew_text(binary_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect text orientation angle using minAreaRect on non-zero text pixels
    and deskew the image back to horizontal.

    Args:
        binary_img: Binarized image where text is black (0) and background is white (255).

    Returns:
        Tuple of (deskewed_binary_image, detected_angle_in_degrees).
    """
    # Invert so text pixels are white (255)
    inv = cv2.bitwise_not(binary_img)

    # Filter out tiny isolated dust speckles to avoid skewing the outer bounding rect
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_inv = np.zeros_like(inv)
    for c in contours:
        if cv2.contourArea(c) > 15:
            cv2.drawContours(clean_inv, [c], -1, 255, -1)

    pts = cv2.findNonZero(clean_inv)
    if pts is None or len(pts) < 50:
        return binary_img, 0.0

    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect

    # OpenCV minAreaRect convention normalization:
    if w < h:
        angle_corr = angle
    else:
        angle_corr = angle + 90.0 if angle < 0 else angle - 90.0

    while angle_corr > 45.0:
        angle_corr -= 90.0
    while angle_corr < -45.0:
        angle_corr += 90.0

    # Rotate if significant skew detected (> 0.25 degrees)
    if abs(angle_corr) >= 0.25:
        (h_img, w_img) = binary_img.shape
        center = (w_img // 2, h_img // 2)
        M = cv2.getRotationMatrix2D(center, angle_corr, 1.0)
        deskewed = cv2.warpAffine(
            binary_img,
            M,
            (w_img, h_img),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return deskewed, angle_corr

    return binary_img, 0.0


def preprocess_image(
    original_rgb: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Runs the complete preprocessing sequence:
    1. Grayscale
    2. Denoise (medianBlur)
    3. Adaptive Threshold (Gaussian adaptive binarization)
    4. Deskew (minAreaRect on text pixels)

    Args:
        original_rgb: RGB NumPy array.

    Returns:
        Tuple of (processed_binary_image, detected_skew_angle).
    """
    # 1. Grayscale
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Denoise (median filter removes scan salt-and-pepper noise while keeping letter edges crisp)
    denoised = cv2.medianBlur(gray, 3)

    # 3. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=17,
        C=9,
    )

    # 4. Deskew
    deskewed, skew_angle = deskew_text(thresh)

    return deskewed, skew_angle


def run_ocr(image: np.ndarray, psm: int = 6) -> str:
    """
    Execute Tesseract OCR on the given image array using specified PSM mode.

    Args:
        image: Grayscale, binary, or RGB image array.
        psm: Page segmentation mode (default: 6 - assume uniform text block).

    Returns:
        Extracted text as string.
    """
    config = f"--psm {psm}"
    return pytesseract.image_to_string(image, config=config)


def process_document(
    source: Union[str, Path, bytes, Image.Image, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float, str, Optional[str]]:
    """
    Full pipeline: loads document, preprocesses image, runs OCR.

    For PDF inputs, also attempts to extract native text via PyMuPDF
    (page.get_text).  If the native text is ≥ 50 characters it is returned
    as the fifth element so the extractor can skip OCR entirely.

    Returns:
        (original_rgb, processed_binary, detected_skew_angle, ocr_text, pdf_native_text)
        pdf_native_text is None for image-only inputs.
    """
    pdf_native_text: Optional[str] = None

    # Attempt native text extraction for PDFs
    is_pdf_path = isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() == ".pdf"
    is_pdf_bytes = isinstance(source, bytes) and source[:4] == b"%PDF"

    if is_pdf_path or is_pdf_bytes:
        try:
            if is_pdf_bytes:
                doc = pymupdf.open(stream=source, filetype="pdf")
            else:
                doc = pymupdf.open(str(source))
            if len(doc) > 0:
                # sort=True gives reading-order layout reconstruction
                native = doc[0].get_text("text", sort=True)
                if len(native.strip()) >= 50:
                    pdf_native_text = native
            doc.close()
        except Exception:
            pass  # silently fall back to OCR path

    original = load_document(source)
    processed, skew_angle = preprocess_image(original)
    ocr_text = run_ocr(processed, psm=6)
    return original, processed, skew_angle, ocr_text, pdf_native_text



if __name__ == "__main__":
    import sys

    # Tiny CLI test that runs on every file in /samples and prints the OCR text
    samples_dir = Path(__file__).resolve().parent / "samples"
    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        sys.exit(1)

    sample_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.pdf")))
    print(f"=== Running preprocess.py CLI Test on {len(sample_files)} sample(s) ===\n")

    for file_path in sample_files:
        print("=" * 70)
        print(f"Processing: {file_path.name}")
        print("=" * 70)
        try:
            orig, proc, angle, text = process_document(file_path)
            print(f"Original shape: {orig.shape} | Processed shape: {proc.shape} | Skew Angle: {angle:.2f}°")
            print("-" * 70)
            print("Extracted OCR Text (PSM 6):")
            print("-" * 70)
            print(text.strip())
            print()
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
