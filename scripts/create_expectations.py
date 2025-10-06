import great_expectations as gx
# Import the required BatchRequest class
from great_expectations.core.batch import BatchRequest
# Great Expectations uses this library for YAML parsing
from ruamel import yaml 


# Get the Data Context
context = gx.get_context()

# --- Part 1: Create the Expectation Suite ---
# Define the batch request dictionary
batch_request_dict = {
    "datasource_name": "olist_datasource",
    "data_connector_name": "default_inferred_data_connector_name",
    "data_asset_name": "olist_orders_dataset",
}

# Define the name for our new suite
expectation_suite_name = "orders.warning"

# Create the expectation suite
context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)

# Get a validator, converting the dictionary to a BatchRequest object
validator = context.get_validator(
    batch_request=BatchRequest(**batch_request_dict), # Use the BatchRequest object here
    expectation_suite_name=expectation_suite_name,
)

print("Adding expectations to the 'orders.warning' suite...")

# Add our specific data quality rules (expectations)
validator.expect_column_values_to_not_be_null("order_id")

validator.expect_column_values_to_be_in_set(
    "order_status",
    ["delivered", "shipped", "invoiced", "processing", "created", "canceled", "unavailable", "approved"]
)

validator.expect_column_values_to_match_strftime_format(
    "order_purchase_timestamp", "%Y-%m-%d %H:%M:%S"
)

# Save the expectation suite to its JSON file
validator.save_expectation_suite(discard_failed_expectations=False)

print(f"Suite '{expectation_suite_name}' was created and saved successfully!")

# --- Part 2: Create the Checkpoint ---
print("\nCreating a checkpoint to validate the orders data...")

# Define the checkpoint configuration in YAML format as a Python string
checkpoint_yaml_string = f"""
name: validate_orders_checkpoint
config_version: 1.0
class_name: SimpleCheckpoint
run_name_template: "%Y%m%d-%H%M%S-validation-run"
validations:
  - batch_request:
      datasource_name: olist_datasource
      data_connector_name: default_inferred_data_connector_name
      data_asset_name: olist_orders_dataset
    expectation_suite_name: {expectation_suite_name}
"""

# Parse the YAML string into a Python dictionary
checkpoint_config = yaml.safe_load(checkpoint_yaml_string)

# Add or update the checkpoint using the parsed dictionary
context.add_or_update_checkpoint(**checkpoint_config)

print("Checkpoint 'validate_orders_checkpoint' was created successfully!")