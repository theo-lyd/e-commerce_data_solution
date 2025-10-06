import os
import pandas as pd
import numpy as np
from collections import Counter
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


def _mode_or_first(series):
    # safe mode: return most common value; if tie or all nan, return first non-null
    vals = series.dropna().tolist()
    if not vals:
        return np.nan
    counts = Counter(vals)
    mode_val, cnt = counts.most_common(1)[0]
    return mode_val

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
    """Transforms raw data into a clean star schema model and preserves payments & reviews dims."""
    print("Transforming data into a star schema...")

    # Convert date columns across all tables (if not already done upstream)
    for df_name, df in dataframes.items():
        for col in df.columns:
            if 'timestamp' in col or '_date' in col:
                dataframes[df_name][col] = pd.to_datetime(df[col], errors='coerce')

    # --- Create Dimension Tables ---

    # 1. dim_customers
    dim_customers = dataframes['customers'].drop_duplicates(subset=['customer_id']).copy()
    print(f"  - Created dim_customers with {len(dim_customers)} unique customers.")

    # 2. dim_products (include translated category)
    products = dataframes['products'].copy()
    translations = dataframes.get('product_category_name_translation', pd.DataFrame())
    if not translations.empty:
        dim_products = pd.merge(products, translations, on='product_category_name', how='left')
    else:
        dim_products = products.copy()
    dim_products = dim_products.drop_duplicates(subset=['product_id'])
    print(f"  - Created dim_products with {len(dim_products)} unique products.")

    # 3. dim_sellers
    dim_sellers = dataframes['sellers'].drop_duplicates(subset=['seller_id']).copy()
    print(f"  - Created dim_sellers with {len(dim_sellers)} unique sellers.")

    # 4. dim_payments: payment-level table (one row per payment event)
    if 'order_payments' in dataframes:
        dp = dataframes['order_payments'].copy()
        # ensure types
        if 'payment_installments' in dp.columns:
            dp['payment_installments'] = pd.to_numeric(dp['payment_installments'], errors='coerce').fillna(0).astype(int)
        dp['payment_value'] = pd.to_numeric(dp['payment_value'], errors='coerce').fillna(0.0)
        # keep raw payments as dim_payments
        dim_payments = dp.drop_duplicates().reset_index(drop=True)
        print(f"  - Created dim_payments with {len(dim_payments)} payment events.")
    else:
        dim_payments = pd.DataFrame(columns=['order_id','payment_sequential','payment_type','payment_installments','payment_value'])
        print("  - order_payments not found; dim_payments empty.")

    # 5. dim_reviews: review-level records (text + dates)
    if 'order_reviews' in dataframes:
        dr = dataframes['order_reviews'].copy()
        # standardize date
        for c in ['review_creation_date', 'review_answer_timestamp']:
            if c in dr.columns:
                dr[c] = pd.to_datetime(dr[c], errors='coerce')
        # keep review text and score
        dim_reviews = dr.drop_duplicates(subset=['review_id']).reset_index(drop=True)
        print(f"  - Created dim_reviews with {len(dim_reviews)} review rows.")
    else:
        dim_reviews = pd.DataFrame(columns=['review_id','order_id','review_score','review_comment_message','review_creation_date'])
        print("  - order_reviews not found; dim_reviews empty.")

    # --- Create the Fact Table ---
    # Start with order_items as base (transaction-level)
    fact_orders = dataframes['order_items'].copy()
    # Include product / price / freight / seller info already present
    # Join with orders to get customer_id and order dates
    orders_cols = ['order_id', 'customer_id', 'order_status',
                   'order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date',
                   'order_estimated_delivery_date']
    orders = dataframes['orders'][orders_cols].copy()
    fact_orders = pd.merge(fact_orders, orders, on='order_id', how='left')

    # --- Payments: create order-level aggregates but preserve primary payment type ---
    # We'll compute per-order aggregates and pick a "primary" payment_type (first by payment_sequential)
    if not dim_payments.empty:
        # Order-level aggregates:
        payments_grp = dim_payments.groupby('order_id').agg(
            payment_value_sum=('payment_value', 'sum'),
            payment_installments_max=('payment_installments', 'max'),
            payment_methods_count=('payment_type', lambda s: s.nunique())
        ).reset_index()

        # Find primary payment type by the smallest payment_sequential per order (first payment event)
        prim = dim_payments.sort_values(['order_id', 'payment_sequential']).drop_duplicates('order_id', keep='first')[['order_id','payment_type','payment_installments']].rename(
            columns={'payment_type':'payment_type_primary','payment_installments':'payment_installments_primary'}
        )

        payments_order = pd.merge(payments_grp, prim, on='order_id', how='left')
        # Merge into fact_orders
        fact_orders = pd.merge(fact_orders, payments_order, on='order_id', how='left')
        print(f"  - Merged payment aggregates into fact_orders (orders with payments: {payments_order['order_id'].nunique()}).")
    else:
        # create empty columns for downstream expectations
        fact_orders['payment_value_sum'] = np.nan
        fact_orders['payment_installments_max'] = np.nan
        fact_orders['payment_methods_count'] = 0
        fact_orders['payment_type_primary'] = np.nan
        fact_orders['payment_installments_primary'] = np.nan
        print("  - No payments to merge into fact_orders.")

    # --- Reviews: attach review-level aggregates and keep dim_reviews for detailed analysis
    if not dim_reviews.empty:
        # per-order review score mean (if multiple reviews per order)
        reviews_grp = dim_reviews.groupby('order_id').agg(
            review_score_mean=('review_score','mean'),
            review_count=('review_id','nunique')
        ).reset_index()
        fact_orders = pd.merge(fact_orders, reviews_grp, on='order_id', how='left')
        print(f"  - Merged review aggregates into fact_orders (orders with reviews: {reviews_grp['order_id'].nunique()}).")
    else:
        fact_orders['review_score_mean'] = np.nan
        fact_orders['review_count'] = 0
        print("  - No reviews found to merge.")

    # --- Feature engineering on fact_orders (example features) ---
    # delivery_time_days: purchase -> delivered to customer
    fact_orders['order_purchase_timestamp'] = pd.to_datetime(fact_orders['order_purchase_timestamp'], errors='coerce')
    fact_orders['order_delivered_customer_date'] = pd.to_datetime(fact_orders['order_delivered_customer_date'], errors='coerce')
    fact_orders['delivery_time_days'] = (fact_orders['order_delivered_customer_date'] - fact_orders['order_purchase_timestamp']).dt.days
    # is_late: delivered after estimated
    fact_orders['order_estimated_delivery_date'] = pd.to_datetime(fact_orders['order_estimated_delivery_date'], errors='coerce')
    fact_orders['is_late'] = (fact_orders['order_delivered_customer_date'] > fact_orders['order_estimated_delivery_date'])

    # freight_ratio: if price exists
    if 'price' in fact_orders.columns and 'freight_value' in fact_orders.columns:
        fact_orders['freight_ratio'] = fact_orders['freight_value'] / fact_orders['price'].replace({0: np.nan})
    else:
        fact_orders['freight_ratio'] = np.nan

    print(f"  - Created fact_orders with {len(fact_orders)} line items (after merges).")

    # Final dictionary of schema tables (including the new dims)
    schema = {
        "fact_orders": fact_orders,
        "dim_customers": dim_customers,
        "dim_products": dim_products,
        "dim_sellers": dim_sellers,
        "dim_payments": dim_payments,   
        "dim_reviews": dim_reviews      
    }

    return schema

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