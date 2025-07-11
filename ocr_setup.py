import pytesseract
import numpy as np
import cv2
import shutil
import os

# A more robust way to configure the Tesseract command.
# It first tries to find Tesseract in the system's PATH.
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("Tesseract executable found in PATH and configured.")
else:
    # If not found in PATH, use the hardcoded path as a fallback.
    print("Tesseract not found in PATH. Using hardcoded path as a fallback.")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

if __name__ == '__main__':
    # 1. Define the project directory and image filename
    # os.path.dirname(__file__) gets the directory of the current script.
    project_dir = os.path.dirname(__file__)
    image_filename = 'in_the_beginning.jpg'
    image_path = os.path.join(project_dir, image_filename)

    try:
        # 2. Check if the file exists before trying to load it
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: Image not found at {image_path}")
        img = cv2.imread(image_path)

        print("Image loaded successfully. Starting preprocessing...")

        # 3. Apply the preprocessing functions in a pipeline
        gray_img = get_grayscale(img)
        thresh_img = thresholding(gray_img)
        # For very noisy images, you might uncomment the next line:
        # final_img = remove_noise(thresh_img)

        # 4. Perform OCR on the preprocessed image
        extracted_text = ocr_core(thresh_img)

        print("\n--- Extracted Text ---")
        print(extracted_text)

    except Exception as e:
        print(e)
