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
    """
    Transforms raw data into a clean star schema model and preserves payments & reviews dims.
    Improvements:
      - canonical key hygiene (cast to str, strip)
      - create dim_orders (one row per order) with canonical order attributes + totals
      - create order-level dims: dim_order_payments, dim_order_reviews (one row per order)
      - keep event-level tables: dim_payment_events, dim_review_events
      - normalize payment_type values (simple mapping + normalization)
      - normalize city names (strip accents) and reconcile city/state using geolocation zip prefix
      - sanity checks / assertions for uniqueness
    """
    import unicodedata
    from collections import Counter

    print("Transforming data into a star schema...")

    def _ensure_datetime(df, col):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    def _to_str_strip(df, col):
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip()
        else:
            df[col] = pd.Series(dtype="string")

    def _strip_accents(s: pd.Series) -> pd.Series:
        """Remove accents and normalize whitespace, keep as lowercase stripped string."""
        def f(val):
            if pd.isna(val):
                return ""
            v = str(val)
            # normalize unicode accents
            v = unicodedata.normalize("NFKD", v)
            v = "".join([c for c in v if not unicodedata.combining(c)])
            # lower, trim, collapse extra spaces and punctuation
            v = v.lower().strip()
            v = pd.Series([v]).str.replace(r'[^\w\s]', ' ', regex=True).iloc[0]
            v = " ".join(v.split())
            return v
        return s.fillna("").map(f).astype('string')

    def _normalize_payment_type(s: pd.Series) -> pd.Series:
        """Lowercase, remove punctuation/extra whitespace and map synonyms to canonical tokens."""
        s = s.fillna("").astype('string').str.strip().str.lower()
        s = s.str.replace(r'[\s\-\_]+', ' ', regex=True).str.strip()
        s = s.str.replace(r'[^\w\s]', '', regex=True).str.strip()

        mapping = {
            'credit card': 'credit_card', 'creditcard': 'credit_card', 'cc': 'credit_card',
            'debit card': 'debit_card', 'debitcard': 'debit_card',
            'voucher': 'voucher', 'boleto': 'boleto', 'paypal': 'paypal', 'pix': 'pix',
            'not defined': 'unknown', 'not_defined': 'unknown', 'not specified': 'unknown',
            'not_specified': 'unknown', 'na': 'unknown', 'none': 'unknown', 'unknown': 'unknown', '': 'unknown'
        }

        def map_val(v):
            if pd.isna(v):
                return 'unknown'
            if v in mapping:
                return mapping[v]
            # handle small synonyms
            if 'card' in v and ('credit' in v or 'cc' in v):
                return 'credit_card'
            if 'card' in v and 'debit' in v:
                return 'debit_card'
            if v == '':
                return 'unknown'
            # default: replace spaces with underscore
            return v.replace(" ", "_")

        return s.map(map_val).astype('string')

    # --- Normalize date-like columns in all source tables ---
    for df_name, df in dataframes.items():
        for col in list(df.columns):
            if 'timestamp' in col or '_date' in col:
                try:
                    dataframes[df_name][col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    dataframes[df_name][col] = pd.to_datetime(df[col].astype(str), errors='coerce')

    # --- 1. dim_customers (key hygiene) ---
    customers = dataframes.get('customers', pd.DataFrame()).copy()
    _to_str_strip(customers, 'customer_id')
    # Normalize customer city/state textual forms (strip accents)
    if 'customer_city' in customers.columns:
        customers['customer_city_norm'] = _strip_accents(customers['customer_city'])
    else:
        customers['customer_city_norm'] = pd.Series(dtype='string')
    if 'customer_state' in customers.columns:
        customers['customer_state'] = customers['customer_state'].astype('string').str.strip().str.upper()
    else:
        customers['customer_state'] = pd.Series(dtype='string')

    dim_customers = customers.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    print(f"  - Created dim_customers with {len(dim_customers)} unique customers.")

    # --- 2. dim_products (existing logic + hygiene) ---
    products = dataframes.get('products', pd.DataFrame()).copy()
    _to_str_strip(products, 'product_id')
    if 'product_category_name' not in products.columns:
        products['product_category_name'] = pd.NA
    products['product_category_name'] = products['product_category_name'].astype('string').str.strip()

    translations = dataframes.get('product_category_name_translation', pd.DataFrame()).copy()
    if not translations.empty:
        key_col = 'product_category_name'
        other_cols = [c for c in translations.columns if c != key_col]
        if other_cols:
            trans_col = other_cols[0]
            translations = translations[[key_col, trans_col]].rename(columns={trans_col: 'product_category_name_translated'})
        else:
            translations = pd.DataFrame()

    if not translations.empty:
        dim_products = pd.merge(products, translations, on='product_category_name', how='left')
    else:
        dim_products = products.copy()
        dim_products['product_category_name_translated'] = pd.NA

    dim_products['product_category_name_translated'] = dim_products['product_category_name_translated'].astype('string')
    dim_products['product_category_name'] = dim_products['product_category_name'].astype('string')

    dim_products['product_category_name_final'] = (
        dim_products['product_category_name_translated']
        .fillna(dim_products['product_category_name'])
        .fillna('UNKNOWN')
        .str.strip()
    )
    dim_products['product_category_name_original'] = dim_products['product_category_name']
    dim_products['product_category_name_missing'] = dim_products['product_category_name_original'].isna()
    dim_products['product_category_name'] = dim_products['product_category_name_final']
    dim_products = dim_products.drop(columns=['product_category_name_final'])
    print(f"  - Created dim_products with {len(dim_products)} unique products. (filled {int(dim_products['product_category_name_missing'].sum())} missing product_category_name)")

    # --- 3. dim_sellers (key hygiene) ---
    sellers = dataframes.get('sellers', pd.DataFrame()).copy()
    _to_str_strip(sellers, 'seller_id')
    if 'seller_city' in sellers.columns:
        sellers['seller_city_norm'] = _strip_accents(sellers['seller_city'])
    else:
        sellers['seller_city_norm'] = pd.Series(dtype='string')
    if 'seller_state' in sellers.columns:
        sellers['seller_state'] = sellers['seller_state'].astype('string').str.strip().str.upper()
    else:
        sellers['seller_state'] = pd.Series(dtype='string')

    dim_sellers = sellers.drop_duplicates(subset=['seller_id']).reset_index(drop=True)
    print(f"  - Created dim_sellers with {len(dim_sellers)} unique sellers.")

    # --- build geolocation mapping to canonical city/state using zip prefix when available ---
    geol = dataframes.get('geolocation', pd.DataFrame()).copy()
    geolocation_map_by_zip = {}
    if not geol.empty:
        # normalize geolocation values
        if 'geolocation_zip_code_prefix' in geol.columns:
            geol['geolocation_zip_code_prefix'] = geol['geolocation_zip_code_prefix'].astype(str)
        if 'geolocation_city' in geol.columns:
            geol['geolocation_city_norm'] = _strip_accents(geol['geolocation_city'])
        else:
            geol['geolocation_city_norm'] = pd.Series(dtype='string')
        if 'geolocation_state' in geol.columns:
            geol['geolocation_state'] = geol['geolocation_state'].astype('string').str.strip().str.upper()
        else:
            geol['geolocation_state'] = pd.Series(dtype='string')

        # prefer the most frequent (city,state) per zip prefix
        if 'geolocation_zip_code_prefix' in geol.columns:
            grouped = (geol.groupby('geolocation_zip_code_prefix').agg(
                city_mode=('geolocation_city_norm', lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna())>0 else ""),
                state_mode=('geolocation_state', lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna())>0 else "")
            ).reset_index())
            for _, row in grouped.iterrows():
                if row['geolocation_zip_code_prefix']:
                    geolocation_map_by_zip[str(row['geolocation_zip_code_prefix'])] = {
                        'city': row['city_mode'],
                        'state': row['state_mode']
                    }

    # --- 4. Event-level payment table (dim_payment_events) + normalize payment types ---
    if 'order_payments' in dataframes and not dataframes['order_payments'].empty:
        dp = dataframes['order_payments'].copy()
        _to_str_strip(dp, 'order_id')
        if 'payment_installments' in dp.columns:
            dp['payment_installments'] = pd.to_numeric(dp['payment_installments'], errors='coerce').fillna(0).astype(int)
        if 'payment_value' in dp.columns:
            dp['payment_value'] = pd.to_numeric(dp['payment_value'], errors='coerce').fillna(0.0)
        if 'payment_type' in dp.columns:
            dp['payment_type'] = _normalize_payment_type(dp['payment_type'])
        else:
            dp['payment_type'] = pd.Series(['unknown'] * len(dp)).astype('string')
        if 'payment_sequential' not in dp.columns:
            dp['payment_sequential'] = pd.RangeIndex(start=1, stop=len(dp)+1)
        # key hygiene on payer/similar columns if present
        if 'payment_type' in dp.columns:
            dp['payment_type'] = dp['payment_type'].astype('string')
        dim_payment_events = dp.drop_duplicates().reset_index(drop=True)
        print(f"  - Created dim_payment_events (event-level payments) with {len(dim_payment_events)} rows.")
    else:
        dim_payment_events = pd.DataFrame(columns=['order_id','payment_sequential','payment_type','payment_installments','payment_value'])
        print("  - No order_payments source found; created empty dim_payment_events.")

    # --- 5. Event-level reviews table (dim_review_events) ---
    if 'order_reviews' in dataframes and not dataframes['order_reviews'].empty:
        dr = dataframes['order_reviews'].copy()
        _to_str_strip(dr, 'order_id')
        if 'review_creation_date' in dr.columns:
            dr['review_creation_date'] = pd.to_datetime(dr['review_creation_date'], errors='coerce')
        if 'review_answer_timestamp' in dr.columns:
            dr['review_answer_timestamp'] = pd.to_datetime(dr['review_answer_timestamp'], errors='coerce')
        dim_review_events = dr.drop_duplicates(subset=['review_id']).reset_index(drop=True)
        print(f"  - Created dim_review_events (event-level reviews) with {len(dim_review_events)} rows.")
    else:
        dim_review_events = pd.DataFrame(columns=['review_id','order_id','review_score','review_comment_message','review_creation_date'])
        print("  - No order_reviews source found; created empty dim_review_events.")

    # --- 6. fact_orders and order-level totals derived from it ---
    fact_orders = dataframes.get('order_items', pd.DataFrame()).copy()
    _to_str_strip(fact_orders, 'order_id'); _to_str_strip(fact_orders, 'product_id'); _to_str_strip(fact_orders, 'seller_id')

    orders_src = dataframes.get('orders', pd.DataFrame()).copy()
    _to_str_strip(orders_src, 'order_id'); _to_str_strip(orders_src, 'customer_id')
    for c in ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']:
        _ensure_datetime(orders_src, c)

    orders_cols = ['order_id', 'customer_id', 'order_status',
                   'order_purchase_timestamp', 'order_approved_at',
                   'order_delivered_carrier_date', 'order_delivered_customer_date',
                   'order_estimated_delivery_date']
    for c in orders_cols:
        if c not in orders_src.columns:
            orders_src[c] = pd.NA

    fact_orders = pd.merge(fact_orders, orders_src[orders_cols], on='order_id', how='left')

    if 'price' in fact_orders.columns:
        fact_orders['price'] = pd.to_numeric(fact_orders['price'], errors='coerce')
    else:
        fact_orders['price'] = np.nan
    if 'freight_value' in fact_orders.columns:
        fact_orders['freight_value'] = pd.to_numeric(fact_orders['freight_value'], errors='coerce')
    else:
        fact_orders['freight_value'] = np.nan

    order_totals = (
        fact_orders.groupby('order_id', as_index=False)
        .agg(order_total_price=('price', 'sum'),
             order_total_freight=('freight_value', 'sum'),
             order_line_items_count=('order_item_id', 'count'))
    )

    # --- 7. dim_order_payments ---
    if not dim_payment_events.empty:
        payments_grp = dim_payment_events.groupby('order_id', as_index=False).agg(
            payment_value_sum=('payment_value', 'sum'),
            payment_installments_max=('payment_installments', 'max'),
            payment_methods_count=('payment_type', lambda s: s.nunique())
        )
        if 'payment_sequential' in dim_payment_events.columns:
            prim = (dim_payment_events.sort_values(['order_id','payment_sequential']).drop_duplicates('order_id',keep='first')
                    [['order_id','payment_type','payment_installments']].rename(columns={'payment_type':'payment_type_primary','payment_installments':'payment_installments_primary'}))
        else:
            prim = (dim_payment_events.groupby('order_id', as_index=False).agg(
                payment_type_primary=('payment_type', lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
                payment_installments_primary=('payment_installments','max')
            ))
        dim_order_payments = pd.merge(payments_grp, prim, on='order_id', how='left').drop_duplicates(subset=['order_id']).reset_index(drop=True)
        print(f"  - Created dim_order_payments with {len(dim_order_payments)} rows.")
    else:
        dim_order_payments = pd.DataFrame(columns=['order_id','payment_value_sum','payment_installments_max','payment_methods_count','payment_type_primary','payment_installments_primary'])
        print("  - No payment events -> created empty dim_order_payments.")

    # --- 8. dim_order_reviews ---
    if not dim_review_events.empty:
        reviews_grp = (dim_review_events.groupby('order_id', as_index=False)
                       .agg(review_score_mean=('review_score','mean'),
                            review_count=('review_id','nunique'),
                            first_review_id=('review_id', lambda s: s.iloc[0] if len(s)>0 else pd.NA),
                            last_review_date=('review_creation_date','max')))
        dim_order_reviews = reviews_grp.drop_duplicates(subset=['order_id']).reset_index(drop=True)
        print(f"  - Created dim_order_reviews with {len(dim_order_reviews)} rows.")
    else:
        dim_order_reviews = pd.DataFrame(columns=['order_id','review_score_mean','review_count','first_review_id','last_review_date'])
        print("  - No review events -> created empty dim_order_reviews.")

    # --- 9. Build dim_orders (canonical) ---
    dim_orders = orders_src.drop_duplicates(subset=['order_id']).reset_index(drop=True)
    _to_str_strip(dim_orders, 'customer_id')

    if not order_totals.empty:
        dim_orders = pd.merge(dim_orders, order_totals, on='order_id', how='left')
    else:
        dim_orders['order_total_price'] = 0.0; dim_orders['order_total_freight'] = 0.0; dim_orders['order_line_items_count'] = 0

    if not dim_order_payments.empty:
        dim_orders = pd.merge(dim_orders, dim_order_payments, on='order_id', how='left')
    else:
        dim_orders['payment_value_sum'] = np.nan; dim_orders['payment_installments_max'] = np.nan; dim_orders['payment_methods_count'] = 0
        dim_orders['payment_type_primary'] = pd.NA; dim_orders['payment_installments_primary'] = np.nan

    if not dim_order_reviews.empty:
        dim_orders = pd.merge(dim_orders, dim_order_reviews, on='order_id', how='left')
    else:
        dim_orders['review_score_mean'] = np.nan; dim_orders['review_count'] = 0; dim_orders['first_review_id'] = pd.NA; dim_orders['last_review_date'] = pd.NaT

    _to_str_strip(dim_orders, 'order_id')

    # --- 10. Reconcile customer/seller city/state using geolocation map_by_zip where possible ---
    def _reconcile_city_state(df, zip_col, city_col, state_col, city_norm_col):
        """Use geolocation_map_by_zip to fix missing/ambiguous city/state using zip prefix where available.
           Returns count of fixes made (city,state)."""
        fixes = 0
        if zip_col in df.columns and df[zip_col].notna().any():
            # convert to string for lookup
            zips = df[zip_col].astype(str).fillna("")
            for idx, z in zips[slice(None)].items():
                if not z:
                    continue
                mapped = geolocation_map_by_zip.get(str(z))
                if mapped:
                    # if state differs or is missing, fix it
                    cur_state = df.at[idx, state_col] if state_col in df.columns else ""
                    cur_state = "" if pd.isna(cur_state) else str(cur_state).strip().upper()
                    if not cur_state or (mapped['state'] and cur_state != mapped['state']):
                        df.at[idx, state_col] = mapped['state']
                        fixes += 1
                    # city
                    cur_city_norm = df.at[idx, city_norm_col] if city_norm_col in df.columns else ""
                    if not cur_city_norm or (mapped['city'] and cur_city_norm != mapped['city']):
                        df.at[idx, city_norm_col] = mapped['city']
                        fixes += 1
        return fixes

    # prepare zip code columns names used in dataset (customer_zip_code_prefix, seller_zip_code_prefix)
    customer_zip_col = 'customer_zip_code_prefix' if 'customer_zip_code_prefix' in dim_customers.columns else None
    seller_zip_col = 'seller_zip_code_prefix' if 'seller_zip_code_prefix' in dim_sellers.columns else None

    cust_fixes = 0
    sell_fixes = 0
    if customer_zip_col:
        dim_customers[customer_zip_col] = dim_customers[customer_zip_col].astype(str)
        cust_fixes = _reconcile_city_state(dim_customers, customer_zip_col, 'customer_city', 'customer_state', 'customer_city_norm')
    if seller_zip_col:
        dim_sellers[seller_zip_col] = dim_sellers[seller_zip_col].astype(str)
        sell_fixes = _reconcile_city_state(dim_sellers, seller_zip_col, 'seller_city', 'seller_state', 'seller_city_norm')

    print(f"  - Reconciled customer city/state using geolocation zip map: {cust_fixes} fixes.")
    print(f"  - Reconciled seller city/state using geolocation zip map: {sell_fixes} fixes.")

    # --- 11. Merge aggregates into fact_orders (denormalize) ---
    if not dim_order_payments.empty:
        fact_orders = pd.merge(fact_orders, dim_order_payments, on='order_id', how='left')
    else:
        fact_orders['payment_value_sum'] = np.nan; fact_orders['payment_installments_max'] = np.nan; fact_orders['payment_methods_count'] = 0
        fact_orders['payment_type_primary'] = pd.NA; fact_orders['payment_installments_primary'] = np.nan

    if not dim_order_reviews.empty:
        fact_orders = pd.merge(fact_orders, dim_order_reviews, on='order_id', how='left')
    else:
        fact_orders['review_score_mean'] = np.nan; fact_orders['review_count'] = 0

    # --- 12. Final key hygiene pass ---
    for df, keys in [(fact_orders, ['order_id','product_id','seller_id']), (dim_orders, ['order_id','customer_id']), (dim_products, ['product_id']), (dim_sellers, ['seller_id']), (dim_customers, ['customer_id'])]:
        for k in keys:
            _to_str_strip(df, k)

    # --- 13. Sanity checks (assertions) ---
    try:
        assert dim_order_payments['order_id'].nunique() == len(dim_order_payments), (
            f"dim_order_payments.order_id must be unique (nunique={dim_order_payments['order_id'].nunique()}, rows={len(dim_order_payments)})"
        )
        assert dim_order_reviews['order_id'].nunique() == len(dim_order_reviews), (
            f"dim_order_reviews.order_id must be unique (nunique={dim_order_reviews['order_id'].nunique()}, rows={len(dim_order_reviews)})"
        )
        assert fact_orders['order_id'].nunique() <= dim_orders['order_id'].nunique(), (
            f"fact_orders has more distinct order_id than dim_orders ({fact_orders['order_id'].nunique()} > {dim_orders['order_id'].nunique()})"
        )
    except AssertionError as ae:
        print("Sanity check failed:", ae)
        print("  - Facts distinct order count:", fact_orders['order_id'].nunique())
        print("  - Dim orders distinct count:", dim_orders['order_id'].nunique())
        print("  - Dim_order_payments distinct / rows:", dim_order_payments['order_id'].nunique(), len(dim_order_payments))
        print("  - Dim_order_reviews distinct / rows:", dim_order_reviews['order_id'].nunique(), len(dim_order_reviews))
        raise

    # --- 14. Build final schema dict ---
    schema = {
        "fact_orders": fact_orders.reset_index(drop=True),
        "dim_customers": dim_customers.reset_index(drop=True),
        "dim_products": dim_products.reset_index(drop=True),
        "dim_sellers": dim_sellers.reset_index(drop=True),
        # order-level unique dims
        "dim_order_payments": dim_order_payments.reset_index(drop=True),
        "dim_order_reviews": dim_order_reviews.reset_index(drop=True),
        # event-level raw tables
        "dim_payment_events": dim_payment_events.reset_index(drop=True),
        "dim_review_events": dim_review_events.reset_index(drop=True),
        # aliases for backward compatibility
        "dim_payments": dim_payment_events.reset_index(drop=True),
        "dim_reviews": dim_review_events.reset_index(drop=True),
        # canonical orders
        "dim_orders": dim_orders.reset_index(drop=True),
    }

    print("Transform complete — returning schema with tables:", ", ".join(schema.keys()))
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