import pandas as pd
import sqlite3
import os
import joblib
import subprocess
import argparse
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_data_from_db(db_path, table_name):
    """Loads data from a specific table in the SQLite database."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT text, language FROM {table_name}", conn)
    return df

def ensure_database_exists(db_path, project_dir):
    """Checks for the database and runs the ingestion script if it's missing."""
    if not os.path.exists(db_path):
        print(f"\nError: Database not found at {db_path}.")
        response = input("Would you like to run 'ingest_data.py' to create it now? (y/n): ")
        if response.lower() == 'y':
            ingest_script_path = os.path.join(project_dir, 'ingest_data.py')
            if not os.path.exists(ingest_script_path):
                print(f"FATAL: 'ingest_data.py' not found at {ingest_script_path}. Cannot proceed.")
                sys.exit(1)

            print("\nRunning data ingestion script...")
            subprocess.run([sys.executable, ingest_script_path], check=True)
            print("Data ingestion complete. Resuming model training...\n")
        else:
            print("Cannot proceed without the database. Exiting.")
            sys.exit(0)

def main(args):
    """Main function to orchestrate the model training pipeline."""
    project_dir = os.path.dirname(__file__)
    data_dir = os.path.join(project_dir, 'data')
    db_path = os.path.join(data_dir, 'geez_data.db')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Ensure data is ready
    ensure_database_exists(db_path, project_dir)

    # 2. Load data (BUG FIX: This now runs after the check, not in an else block)
    print("Loading training and testing data from the database...")
    train_df = load_data_from_db(db_path, 'train')
    test_df = load_data_from_db(db_path, 'test')

    X_train, y_train = train_df['text'], train_df['language']
    X_test, y_test = test_df['text'], test_df['language']
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} testing samples.")

    # 3. Create a Model Pipeline
    # A pipeline bundles a vectorizer and a classifier. This is a best practice.
    # TfidfVectorizer: Converts text into numerical features.
    # MultinomialNB: A simple and effective classifier for text.
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('classifier', MultinomialNB())
    ])

    # 4. Train the Model
    print("\nTraining the language identification model...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate the Model
    print("\nEvaluating the model on the test set...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # 6. Save the Trained Model
    model_path = os.path.join(models_dir, args.output_filename)
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel pipeline saved to: {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a language identification model for Ge'ez-based languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-o', '--output-filename',
        type=str,
        default='lid_model.joblib',
        help='Filename for the saved model pipeline in the models/ directory.'
    )
    args = parser.parse_args()
    main(args)
