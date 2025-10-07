# scripts/ge_inspect.py
import great_expectations as ge
import os

# Dynamically get absolute path to great_expectations directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GE_ROOT = os.path.join(PROJECT_ROOT, 'great_expectations')

ctx = ge.get_context(context_root_dir=GE_ROOT)

print("=== Available expectation suites ===")
suites = ctx.list_expectation_suites()
for s in suites:
    print(" -", s.expectation_suite_name)

print("\n=== Available data assets for datasource 'olist_datasource' ===")
assets = ctx.get_available_data_asset_names()
if "olist_datasource" in assets:
    print(assets["olist_datasource"])
else:
    print("⚠ No datasource named 'olist_datasource' found. Check your great_expectations.yml configuration.")
