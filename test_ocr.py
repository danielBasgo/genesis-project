import os
import argparse
import logging
import cv2
from geez_toolkit.utils import preprocess_image, ocr_core

def test_ocr_on_directory(image_dir, lang, psm, output_dir=None):
    """
    Runs OCR on all supported image files in a given directory.
    """
    if not os.path.isdir(image_dir):
        logging.error(f"Provided path is not a valid directory: {image_dir}")
        return

    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    logging.info(f"Starting OCR test on directory: {image_dir}")

    for image_name in sorted(os.listdir(image_dir)):
        if not image_name.lower().endswith(supported_extensions):
            continue

        image_path = os.path.join(image_dir, image_name)
        try:
            logging.info(f"--- Processing: {image_name} ---")
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Could not read image {image_name}. Skipping.")
                continue

            processed_img = preprocess_image(img)
            custom_config = f'--psm {psm}'
            text = ocr_core(processed_img, lang=lang, config=custom_config)

            # If an output directory is specified, save the result to a file.
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_filename = os.path.splitext(image_name)[0] + '.txt'
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"Result for {image_name} saved to {output_path}")
            else:
                # Otherwise, print to the console.
                print(f"\n--- OCR Output for {image_name} ---")
                print(text if text.strip() else "[No text found]")
                print("--------------------------------------\n")

        except Exception as e:
            logging.error(f"An error occurred processing {image_name}: {e}", exc_info=True)

    logging.info("OCR testing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test Tesseract OCR on a directory of images for a specific language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image_dir', help="Path to the directory containing images to test.")
    parser.add_argument('-l', '--lang', default='gez', help="Tesseract language code to use (e.g., 'gez', 'amh').")
    parser.add_argument('--psm', type=int, default=6, help="Tesseract Page Segmentation Mode (PSM).")
    parser.add_argument('-o', '--output-dir', help="Optional. Directory to save OCR results as .txt files.")
    args = parser.parse_args()

    test_ocr_on_directory(args.image_dir, args.lang, args.psm, args.output_dir)
