import os
import sys
import shutil
import cv2
import pytesseract

def configure_tesseract():
    """
    Finds and configures the Tesseract executable path. Exits if not found.
    This function is called automatically when the module is imported.
    """
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return  # Success

    # Fallback for Windows if not in PATH
    windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
        return  # Success

    print("FATAL: Tesseract executable not found in PATH or at the default Windows location.")
    print("Please install Tesseract and ensure it's in your system's PATH.")
    sys.exit(1)

# --- Image Preprocessing Functions ---

def get_grayscale(image):
    """Converts an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    """Removes noise from an image using a median blur."""
    return cv2.medianBlur(image, 5)

def thresholding(image):
    """Applies binary and Otsu's thresholding to an image."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess_image(image):
    """Applies a standard preprocessing pipeline to an image for OCR."""
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    return thresh

def ocr_core(img, lang='eng', config=''):
    """Performs OCR on an image, extracting text."""
    return pytesseract.image_to_string(img, lang=lang, config=config)

# --- Module-level setup ---
configure_tesseract()

