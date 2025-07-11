import os
import cv2
import joblib
import argparse
import sys
from utils import preprocess_image, ocr_core

# --- Main Application Logic ---

def predict_from_image(image_path, model, ocr_lang, psm):
    """
    Orchestrates the full pipeline: OCR -> Language Identification.
    """
    # 1. Load and preprocess the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Error: Could not read image file at {image_path}")

    print(f"Processing image: {os.path.basename(image_path)}...")
    processed_image = preprocess_image(image)

    # 2. Perform OCR to extract text
    print("Extracting text via OCR...")
    ocr_config = f'--psm {psm}'
    extracted_text = ocr_core(processed_image, lang=ocr_lang, config=ocr_config).strip()

    if not extracted_text:
        print("\nOCR did not find any text in the image.")
        return

    print("\n--- Extracted Text ---")
    print(extracted_text)

    # 3. Predict the language of the extracted text
    print("\n--- Language Identification ---")
    predicted_language = model.predict([extracted_text])
    # The model returns an array, so we get the first element.
    print(f"Predicted Language: {predicted_language[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract text from an image using OCR and predict its language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image_path', help="Path to the input image file.")
    parser.add_argument(
        '-m', '--model-path',
        default=os.path.join('models', 'lid_model.joblib'),
        help="Path to the trained language identification model file."
    )
    parser.add_argument(
        '--ocr-lang', default='amh', help="Tesseract language model to use for the initial OCR step."
    )
    parser.add_argument(
        '--psm', type=int, default=6, help="Tesseract Page Segmentation Mode (PSM) for OCR."
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at '{args.model_path}'. Please run 'train_model.py' first.")
        
        lid_model = joblib.load(args.model_path)
        predict_from_image(args.image_path, lid_model, args.ocr_lang, args.psm)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)