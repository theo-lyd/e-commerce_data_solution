import os
import pandas as pd
import json
import shutil
import subprocess

# Define file paths
DATA_DIR = '../data'
OUTPUT_DIR = '../output'
TEMP_DIR = os.path.join(OUTPUT_DIR, '_tmp')
WATERMARK_FILE = os.path.join(OUTPUT_DIR, '_watermark.json')

def extract_data():
    """Reads all CSV files from the data directory into a dictionary of DataFrames."""
    print("Starting data extraction...")
    
    # In the extract_data function:
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    dataframes = {}
    for file in all_files:
        table_name = file.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')
        file_path = os.path.join(DATA_DIR, file)
        dataframes[table_name] = pd.read_csv(file_path)
        print(f"  - Loaded {table_name} with {len(dataframes[table_name])} rows.")

    print("Data extraction complete.")
    return dataframes

def transform_data(dataframes):
    """Cleans, merges, and engineers features from the extracted data."""
    print("Starting data transformation...")
    
    # In the transform_data function:
    if not dataframes:
        print("No dataframes to transform. Aborting.")
        return None

    # 1. Cleaning: Convert date columns to datetime objects
    for df_name, df in dataframes.items():
        for col in df.columns:
            if 'timestamp' in col or '_date' in col:
                dataframes[df_name][col] = pd.to_datetime(df[col], errors='coerce')

    print("  - Converted date columns to datetime.")

    # 2. Joining (Merging) DataFrames based on the schema
    # Start with orders as the central table
    df = dataframes['orders'].copy()

    # Merge other tables
    df = pd.merge(df, dataframes['order_payments'], on='order_id', how='left')
    df = pd.merge(df, dataframes['order_reviews'], on='order_id', how='left')
    df = pd.merge(df, dataframes['customers'], on='customer_id', how='left')

    # Order items needs to be joined with products and sellers first
    order_items = dataframes['order_items'].copy()
    order_items = pd.merge(order_items, dataframes['products'], on='product_id', how='left')
    order_items = pd.merge(order_items, dataframes['sellers'], on='seller_id', how='left')

    # Now merge the enriched order_items table
    df = pd.merge(df, order_items, on='order_id', how='left')

    # Finally, merge the category name translation
    df = pd.merge(df, dataframes['product_category_name_translation'], on='product_category_name', how='left')

    print(f"  - Merged all dataframes into a single table with {len(df)} rows.")

    print("  - Re-asserting datetime types after merges...")
    date_cols_for_calc = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'order_delivered_carrier_date',
        'order_approved_at'
    ]
    for col in date_cols_for_calc:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Feature Engineering
    # Calculate delivery time in days
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 60 * 60)

    # Flag for late deliveries
    df['is_late'] = df['order_delivered_customer_date'] > df['order_estimated_delivery_date']

    # Calculate seller handling time in days
    df['seller_handling_time_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / (24 * 60 * 60)

    # Calculate freight ratio
    df['freight_ratio'] = df['freight_value'] / df['price']

    print("  - Engineered new features: delivery_time_days, is_late, seller_handling_time_days, freight_ratio.")

    print("Data transformation complete.")
    return df

def load_data(df):
    """Saves the transformed DataFrame as a partitioned Parquet file idempotently."""
    print("Starting data loading...")
    
    # In the load_data function:
    if df is None or df.empty:
        print("No data to load. Aborting.")
        return

    # Ensure the temporary directory exists and is empty
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # Add year and month columns for partitioning
    df['year'] = df['order_purchase_timestamp'].dt.year
    df['month'] = df['order_purchase_timestamp'].dt.month

    # Save to temporary partitioned Parquet files
    df.to_parquet(
        TEMP_DIR,
        partition_cols=['year', 'month'],
        engine='pyarrow',
        compression='snappy'
    )
    print(f"  - Saved data temporarily to {TEMP_DIR}")

    # Atomic move: remove old data and replace with new
    final_output_path = os.path.join(OUTPUT_DIR, 'olist_master.parquet')
    if os.path.exists(final_output_path):
        shutil.rmtree(final_output_path)

    os.rename(TEMP_DIR, final_output_path)
    print(f"  - Atomically moved data to {final_output_path}")

    # Update watermark
    last_processed_date = df['order_purchase_timestamp'].max().isoformat()
    with open(WATERMARK_FILE, 'w') as f:
        json.dump({'last_processed_date': last_processed_date}, f)

    print(f"  - Updated watermark with last date: {last_processed_date}")
    print("Data loading complete.")

def main():
    """Main function to orchestrate the ETL process."""
    print("ETL process started.")

    # Step 1: Extract
    extracted_data = extract_data()

    # New Step: Validate Source Data
    print("Validating source data...")
    validation_process = subprocess.run(
        ["python", "validate_data.py", "validate_orders_checkpoint"], # Use the correct checkpoint name
        capture_output=True, text=True
    )
    print(validation_process.stdout)
    if validation_process.returncode != 0:
        print("Source data validation failed. Aborting ETL process.")
        print(validation_process.stderr)
        return # Stop the pipeline

    # Step 2: Transform
    transformed_df = transform_data(extracted_data)

    # Step 3: Load
    load_data(transformed_df)

    print("ETL process completed successfully.")

if __name__ == "__main__":
    main()