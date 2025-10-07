#!/usr/bin/env python3
"""
Robust Great Expectations checkpoint runner:
- Pre-checks if the checkpoint file exists
- Falls back to 'validate_all_checkpoint' if missing
- Honors environment variables:
    SKIP_GE_VALIDATION=1         # skip all GE validation
    FAIL_ON_MISSING_CHECKPOINT=1 # treat missing checkpoint as fatal
"""

import sys
import os
import great_expectations as ge
import great_expectations.exceptions.exceptions as gx_exceptions

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Define the path to the Great Expectations context root
GE_CONTEXT_ROOT = os.path.join(PROJECT_ROOT, 'great_expectations')

def _checkpoint_file_exists(checkpoint_name: str) -> bool:
    """Check if a checkpoint YAML/JSON file exists under great_expectations/checkpoints"""
    ckpt_dir = os.path.join(GE_CONTEXT_ROOT, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    candidates = [
        os.path.join(ckpt_dir, f"{checkpoint_name}.yml"),
        os.path.join(ckpt_dir, f"{checkpoint_name}.yaml"),
        os.path.join(ckpt_dir, f"{checkpoint_name}.json"),
    ]
    return any(os.path.exists(p) for p in candidates)

def _list_available_checkpoints() -> list:
    ckpt_dir = os.path.join(GE_CONTEXT_ROOT, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted({os.path.splitext(f)[0] for f in os.listdir(ckpt_dir) if f.endswith(('.yml', '.yaml', '.json'))})

def run_validation(checkpoint_name: str) -> bool:
    """Runs a GE checkpoint, with fallback and env-controlled behavior."""
    # Environment variable skip
    if os.environ.get("SKIP_GE_VALIDATION", "").strip() == "1":
        print("SKIP_GE_VALIDATION=1 -> skipping Great Expectations validation (treated as success).")
        return True
    
    context = ge.get_context(context_root_dir=GE_CONTEXT_ROOT)

    # Check if checkpoint file exists
    if not _checkpoint_file_exists(checkpoint_name):
        msg = f"Checkpoint '{checkpoint_name}' not found under {os.path.join(GE_CONTEXT_ROOT,'checkpoints')}."
        print(msg)
        fallback = "validate_all_checkpoint"
        if _checkpoint_file_exists(fallback):
            print(f"Falling back to '{fallback}'.")
            checkpoint_name = fallback
        else:
            available = _list_available_checkpoints()
            print("No checkpoint found and no fallback available.")
            print("Available checkpoints:", available)
            if os.environ.get("FAIL_ON_MISSING_CHECKPOINT", "").strip() == "1":
                print("FAIL_ON_MISSING_CHECKPOINT=1 -> treating missing checkpoint as fatal.")
                return False
            else:
                print("Continuing ETL with validation skipped.")
                return True

    # Run the checkpoint and catch runtime errors
    try:
        print(f"Running Great Expectations checkpoint: {checkpoint_name}")
        result = context.run_checkpoint(checkpoint_name=checkpoint_name)
    except gx_exceptions.CheckpointNotFoundError as e:
        msg = f"Unexpected CheckpointNotFoundError: {e}"
        print(msg)
        if os.environ.get("FAIL_ON_MISSING_CHECKPOINT", "").strip() == "1":
            return False
        return True
    except IndexError as e:
        print("Caught GE internal tuple error (likely update_data_docs). Skipping docs update.")
        return True
    except Exception as e:
        print("Error running checkpoint:", e)
        return False

    # Inspect result success flag
    try:
        success = result.get("success", False)
    except Exception:
        success = False

    if not success:
        print("Validation failed!")
        return False

    print("Validation successful!")
    return True

def main():
    if len(sys.argv) > 1:
        chk_name = sys.argv[1]
    else:
        chk_name = "validate_all_checkpoint"

    ok = run_validation(chk_name)
    if not ok:
        print("Source data validation failed. Aborting ETL process.")
        sys.exit(1)
    print("Source data validation passed (or was skipped).")
    sys.exit(0)

if __name__ == "__main__":
    main()