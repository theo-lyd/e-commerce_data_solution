# scripts/ge_create_baseline_suites.py
import great_expectations as ge
import os
from great_expectations.core.batch import BatchRequest

# Dynamically get absolute path to great_expectations directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GE_ROOT = os.path.join(PROJECT_ROOT, 'great_expectations')

DATASOURCE = "olist_datasource"
DATA_CONNECTOR = "default_inferred_data_connector_name"

ctx = ge.get_context(context_root_dir=GE_ROOT)

# Each entry specifies table-specific rules (non-null key columns, numeric columns ≥0)
ASSETS = {
    "fact_orders": {
        "not_null_cols": ["order_id", "product_id", "customer_id", "order_purchase_timestamp"],
        "non_negative_cols": ["price", "freight_value", "payment_value"]
    },
    "dim_customers": {
        "not_null_cols": ["customer_id", "customer_unique_id"],
        "non_negative_cols": []
    },
    "dim_products": {
        "not_null_cols": ["product_id", "product_category_name"],
        "non_negative_cols": ["product_photos_qty"]
    },
    "dim_sellers": {
        "not_null_cols": ["seller_id"],
        "non_negative_cols": []
    },
    "dim_reviews": {
        "not_null_cols": ["review_id", "order_id", "review_score"],
        "non_negative_cols": ["review_score"]
    },
    "dim_payments": {
        "not_null_cols": ["order_id", "payment_type", "payment_value"],
        "non_negative_cols": ["payment_value", "payment_installments"]
    },
}

def create_suite_for_asset(asset_name, params):
    suite_name = f"{asset_name}.warning"
    print(f"\n--- Processing asset: {asset_name} -> suite: {suite_name}")

    # Create suite if not exists
    try:
        ctx.create_expectation_suite(expectation_suite_name=suite_name, overwrite_existing=False)
        print(f"  ✓ Created expectation suite: {suite_name}")
    except Exception as e:
        print(f"  ⚠ Suite already exists or cannot create: {e}")

    # Build a batch request
    batch_request = BatchRequest(
        datasource_name=DATASOURCE,
        data_connector_name=DATA_CONNECTOR,
        data_asset_name=asset_name,
    )

    try:
        validator = ctx.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
    except Exception as e:
        print(f"  ❌ ERROR obtaining validator for {asset_name}: {e}")
        return

    # Add basic expectations
    try:
        validator.expect_table_row_count_to_be_between(min_value=1, max_value=None)
        print("   - expect_table_row_count_to_be_between(>=1)")

        for col in params.get("not_null_cols", []):
            if col in validator.active_batch.data.dataframe.columns:
                validator.expect_column_values_to_not_be_null(col)
                print(f"   - expect_column_values_to_not_be_null({col})")
            else:
                print(f"   - ⚠ Column '{col}' not found, skipping not-null check")

        for col in params.get("non_negative_cols", []):
            if col in validator.active_batch.data.dataframe.columns:
                validator.expect_column_values_to_be_between(col, min_value=0, max_value=None)
                print(f"   - expect_column_values_to_be_between({col} ≥ 0)")
            else:
                print(f"   - ⚠ Column '{col}' not found, skipping non-negative check")

        validator.save_expectation_suite(discard_failed_expectations=False)
        print(f"  ✓ Saved expectation suite: {suite_name}")

    except Exception as e:
        print(f"  ❌ ERROR adding expectations for {asset_name}: {e}")

if __name__ == "__main__":
    print("=== Creating baseline expectation suites ===")
    for asset, params in ASSETS.items():
        create_suite_for_asset(asset, params)
    print("\n✅ Baseline suite creation complete. You can now run:")
    print("   python scripts/validate_data.py validate_all_checkpoint")
