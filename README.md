# Ge'ez AI Toolkit: OCR and Language Identification

This project provides a suite of tools for working with Ge'ez-based languages, focusing on Optical Character Recognition (OCR) and Language Identification (LID). It includes scripts for data ingestion, model training, and text extraction from images, built upon the Tesseract OCR engine and scikit-learn.

## Features

*   **OCR for Ge'ez Script:** Extracts text from images using a pre-trained Tesseract model. Includes image preprocessing steps to improve accuracy.
*   **Language Identification Model:** Trains a Naive Bayes classifier on the `GeezSwitch` dataset to distinguish between Amharic, Ge'ez, Tigre, and Tigrinya.
*   **Structured Data Pipeline:** Ingests raw `.tsv` data into a structured and efficient SQLite database.
*   **Modular and Reproducible:** The project is organized into clear, single-purpose scripts with a version-pinned `requirements.txt` file for a stable environment.

## Project Motivation: Linguistic Context

This project is particularly effective for anyone with a background in Ethiosemitic languages like Tigrinya or Amharic. The close relationship between these languages and Ge'ez provides a significant advantage in data curation, model evaluation, and overall development.

### Linguistic Proximity

*   **Shared Ancestry:** Tigrinya and Amharic are direct descendants of Ge'ez, belonging to the Ethiosemitic branch of the Semitic language family.
*   **Lexical Similarities:** A substantial amount of shared vocabulary exists, making it easier to recognize and understand Ge'ez words.
*   **Grammatical Similarities:** While Ge'ez grammar is more archaic, the fundamental structures, word order, and morphological patterns (like root-and-pattern) will be familiar.

### Script Familiarity

*   **Shared Abugida:** Both Tigrinya and Ge'ez use the same Ge'ez abugida (fidel script). This provides a massive head start in reading the source material and evaluating OCR output.

### Data Annotation and Curation

A background in a related language is invaluable for every stage of the data pipeline:

*   **Identifying High-Quality Data:** Better discern which datasets or scanned texts are valuable and accurate.
*   **Manual Correction:** More easily identify and correct OCR errors, which is crucial for low-resource languages.
*   **Creating Parallel Corpora:** Linguistic intuition is a powerful tool for creating or verifying parallel sentences for machine translation tasks.
*   **Morphological Analysis:** Understanding Tigrinya's morphology provides insights into how Ge'ez words are formed and inflected, aiding in tasks like part-of-speech tagging or lemmatization.

### Transfer Learning Intuition

Knowing the linguistic relationship helps in understanding why transfer learning from Amharic or other Ethiosemitic languages is a strong strategy. It reinforces the idea that these languages share underlying features that an AI model can leverage.

## Project Structure

```text
.
├── data/
│   ├── GeezSwitch-data/  # Raw dataset
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── train.tsv
│   └── geez_data.db      # (Created by ingest_data.py)
├── models/
│   └── lid_model.joblib  # (Created by train_model.py)
├── utils.py              # Shared utility functions for OCR
├── explore_data.py       # Utility to inspect the dataset
├── ingest_data.py        # Script to populate the SQLite database
├── ocr_setup.py          # Script to perform OCR on an image
├── train_model.py        # Script to train the language ID model
├── predict_from_image.py # Script to run the full OCR -> LID pipeline
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── Dockerfile            # For containerized setup
├── .gitignore            # Specifies untracked files to ignore
└── .dockerignore         # Specifies files to exclude from Docker image
└── Dockerfile            # For containerized setup
```

## Setup and Installation

#### 1. Clone the Repository

```sh
git clone https://github.com/danielBasgo/genesis-project.git
cd genesis-project
```

#### 2. Prerequisites

You must have the **Tesseract OCR engine** installed on your system.

*   **Windows:** Download the installer from Tesseract at UB Mannheim. During installation, make sure to select and install the language packs you need (e.g., Amharic `amh`, Tigrinya `tir`).
*   **macOS:** `brew install tesseract tesseract-lang`
*   **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr tesseract-ocr-all`

#### 3. Set Up Python Environment

It is highly recommended to use a virtual environment.

```sh
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

## Usage

The scripts are designed to be run in a specific order.

### Step 1: Ingest the Data

This script reads the raw `.tsv` files and creates the `geez_data.db` database. This only needs to be run once.

```sh
python ingest_data.py
```

### Step 2: Train the Language Identification Model

This script uses the data from the database to train a classifier and saves the resulting model to the `models/` directory.

```sh
python train_model.py
```
You can specify a different output filename for the model:
```sh
python train_model.py --output-filename my_experiment.joblib
```

### Step 3: Perform OCR on an Image

This script demonstrates how to extract text from an image. You will need to edit the `image_filename` variable within `ocr_setup.py` to point to your target image.

```sh
python ocr_setup.py
```

### (Optional) Explore the Dataset

To get a quick summary of the training data, you can run the exploration script.

```sh
python explore_data.py
```