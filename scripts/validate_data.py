import sys
import great_expectations as ge

def run_validation(checkpoint_name: str) -> bool:
    """Runs a Great Expectations checkpoint."""
    print(f"Running Great Expectations checkpoint: {checkpoint_name}")
    context = ge.get_context()

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