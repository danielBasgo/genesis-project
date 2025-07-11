import pandas as pd
import sqlite3
import os

def ingest_data_to_db(conn, tsv_file_path, table_name):
    """
    Reads data from a TSV file and ingests it into a specified
    table in the SQLite database.

    Args:
        conn: An active sqlite3 database connection.
        tsv_file_path (str): The path to the .tsv file.
        table_name (str): The name of the table to create in the database.
    """
    if not os.path.exists(tsv_file_path):
        print(f"Warning: File not found at {tsv_file_path}. Skipping.")
        return

    print(f"Reading data from {os.path.basename(tsv_file_path)}...")
    # Load the dataset using pandas, assigning column names
    df = pd.read_csv(tsv_file_path, sep='\t', header=None, names=['id', 'language', 'text'])

    print(f"Writing data to '{table_name}' table...")
    # Use pandas' to_sql method to write the data.
    # 'if_exists="replace"' will overwrite the table if it already exists,
    # making this script safe to re-run.
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Successfully ingested {len(df)} rows into '{table_name}'.")

if __name__ == '__main__':
    project_dir = os.path.dirname(__file__)
    data_dir = os.path.join(project_dir, 'data')
    db_path = os.path.join(data_dir, 'geez_data.db')

    # The data files to process
    datasets = {
        'train': os.path.join(data_dir, 'train.tsv'),
        'dev': os.path.join(data_dir, 'dev.tsv'),
        'test': os.path.join(data_dir, 'test.tsv')
    }

    with sqlite3.connect(db_path) as conn:
        print(f"Database created/connected at: {db_path}")
        for table_name, file_path in datasets.items():
            ingest_data_to_db(conn, file_path, table_name)
    print("\nData ingestion complete.")
