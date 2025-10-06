import sys
import os
import great_expectations as ge

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the path to the Great Expectations context root
GE_CONTEXT_ROOT = os.path.join(PROJECT_ROOT, 'great_expectations')

def run_validation(checkpoint_name: str) -> bool:
    """Runs a Great Expectations checkpoint."""
    print(f"Running Great Expectations checkpoint: {checkpoint_name}")
    # Get the context from the explicit root directory
    context = ge.get_context(context_root_dir=GE_CONTEXT_ROOT)

    result = context.run_checkpoint(checkpoint_name=checkpoint_name)

    if not result["success"]:
        print("Validation failed!")
        return False

    print("Validation successful!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        chk_name = sys.argv[1]
        if not run_validation(chk_name):
            sys.exit(1) # Exit with a non-zero status code for failure
    else:
        print("Error: Please provide a checkpoint name to run.")
        sys.exit(2)