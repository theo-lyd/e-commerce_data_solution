import os
import pandas as pd
import json
import shutil
import subprocess

# --- ROBUST PATH DEFINITION ---
# Get the directory of the currently running script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# Go one level up to get the project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# Define all other paths relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
VALIDATION_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'validate_data.py')


def extract_data():
    """Reads all CSV files from the data directory into a dictionary of DataFrames."""
    print("Starting data extraction...")
    
    # In the extract_data function:
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    dataframes = {}
    for file in all_files:
        clean_name = file.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')
        if 'product_category_name_translation' in clean_name:
            clean_name = 'product_category_name_translation'
        file_path = os.path.join(DATA_DIR, file)
        dataframes[clean_name] = pd.read_csv(file_path)
        print(f"  - Loaded {clean_name} with {len(dataframes[clean_name])} rows.")

    print("Data extraction complete.")
    return dataframes

def transform_and_create_schema(dataframes):
    """Transforms raw data into a clean star schema model."""
    print("Transforming data into a star schema...")

    # --- Data Cleaning and Preparation ---
    # Convert date columns across all tables
    for df_name, df in dataframes.items():
        for col in df.columns:
            if 'timestamp' in col or '_date' in col:
                dataframes[df_name][col] = pd.to_datetime(df[col], errors='coerce')

    # --- Create Dimension Tables ---

    # 1. dim_customers
    dim_customers = dataframes['customers'].drop_duplicates(subset=['customer_id']).copy()
    print(f"  - Created dim_customers with {len(dim_customers)} unique customers.")

    # 2. dim_products
    products = dataframes['products'].copy()
    translations = dataframes['product_category_name_translation'].copy()
    dim_products = pd.merge(products, translations, on='product_category_name', how='left')
    dim_products = dim_products.drop_duplicates(subset=['product_id'])
    print(f"  - Created dim_products with {len(dim_products)} unique products.")

    # 3. dim_sellers
    dim_sellers = dataframes['sellers'].drop_duplicates(subset=['seller_id']).copy()
    print(f"  - Created dim_sellers with {len(dim_sellers)} unique sellers.")

    # --- Create the Fact Table ---
    # Start with order_items as the base for the fact table, as it has the core transaction data
    fact_orders = dataframes['order_items'].copy()

    # Join with orders to get customer_id and order date information
    orders = dataframes['orders'][['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']]
    fact_orders = pd.merge(fact_orders, orders, on='order_id', how='left')

    # Join with order_payments to get payment info
    payments = dataframes['order_payments'].groupby('order_id').agg(
        payment_sequential=('payment_sequential', 'max'),
        payment_installments=('payment_installments', 'sum'),
        payment_value=('payment_value', 'sum')
    ).reset_index()
    fact_orders = pd.merge(fact_orders, payments, on='order_id', how='left')

    # Join with order_reviews to get review scores
    reviews = dataframes['order_reviews'][['order_id', 'review_score']].groupby('order_id').agg(review_score=('review_score', 'mean')).reset_index()
    fact_orders = pd.merge(fact_orders, reviews, on='order_id', how='left')
    
    print(f"  - Created fact_orders with {len(fact_orders)} line items.")

    return {
        "fact_orders": fact_orders,
        "dim_customers": dim_customers,
        "dim_products": dim_products,
        "dim_sellers": dim_sellers
    }

def load_schema_tables(schema_tables):
    """Saves each table in the star schema as a separate Parquet file."""
    print("Loading schema tables to Parquet files...")
    # Clear the output directory for a clean load
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for name, df in schema_tables.items():
        file_path = os.path.join(OUTPUT_DIR, f'{name}.parquet')
        df.to_parquet(file_path, index=False)
        print(f"  - Saved {name} to {file_path}")
    print("Data loading complete.")

def main():
    """Main function to orchestrate the ETL process."""
    print("ETL process started.")
    extracted_data = extract_data()

    # The validation step can remain the same as it checks the raw source data
    print("Validating source data...")
    validation_process = subprocess.run(
        ["python", VALIDATION_SCRIPT_PATH, "validate_orders_checkpoint"],
        capture_output=True, text=True
    )
    print(validation_process.stdout)
    if validation_process.returncode != 0:
        print("Source data validation failed. Aborting ETL process.")
        print(validation_process.stderr)
        return

    schema_tables = transform_and_create_schema(extracted_data)
    load_schema_tables(schema_tables)
    print("ETL process completed successfully.")

if __name__ == "__main__":
    main()