import cv2
import argparse
import os
import logging
from geez_toolkit.utils import preprocess_image, ocr_core

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform OCR on an image file using Tesseract.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image_path', help="Path to the image file.")
    parser.add_argument(
        '-l', '--lang',
        default='amh',
        help="Tesseract language code to use (e.g., 'eng', 'amh', 'tir')."
    )
    parser.add_argument(
        '--psm',
        type=int,
        default=6,
        help="Tesseract Page Segmentation Mode (PSM)."
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Error: Image not found at {args.image_path}")
        
        img = cv2.imread(args.image_path)
        logging.info(f"Image '{os.path.basename(args.image_path)}' loaded successfully. Starting OCR...")

        processed_img = preprocess_image(img)
        custom_config = f'--psm {args.psm}'
        extracted_text = ocr_core(processed_img, lang=args.lang, config=custom_config)

        # The result is the primary output, so we use print() for it.
        print("\n--- OCR Result ---")
        print(extracted_text)

    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
