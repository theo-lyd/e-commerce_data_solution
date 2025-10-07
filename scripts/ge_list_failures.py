#!/usr/bin/env python3
# scripts/ge_inspect_full_checkpoint.py
"""
1) Scan all saved validation JSONs and print failures (if any).
2) Programmatically run the checkpoint "validate_all_checkpoint" and
   print *all* failing expectations from the run_results.
3) Save the checkpoint run output to great_expectations/uncommitted/validations/auto_run_output_<ts>.json

Run:
  python scripts/ge_inspect_full_checkpoint.py
"""

import json
import glob
import os
import sys
from datetime import datetime
from great_expectations.data_context import DataContext

def scan_saved_validations(context):
    repo_root = context.root_directory
    candidates = []
    cand_dirs = [
        os.path.join(repo_root, "great_expectations", "uncommitted", "validations"),
        os.path.join(repo_root, "uncommitted", "validations"),
        os.path.join(repo_root, "great_expectations", "validations"),
    ]
    for d in cand_dirs:
        if os.path.isdir(d):
            candidates += glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    print(f"Found {len(candidates)} saved validation JSON(s) in: {', '.join([p for p in cand_dirs if os.path.isdir(p)])}\n")
    any_failures = False
    for p in candidates:
        try:
            j = json.load(open(p, "r", encoding="utf-8"))
        except Exception as e:
            print(f"  Error reading {p}: {e}")
            continue
        fails = extract_failures_from_validation_json(j)
        if fails:
            any_failures = True
            print(f"Failures in saved validation file: {p}")
            print_failures(fails)
    if not any_failures:
        print("No failures found in any saved validation JSONs.\n")

def extract_failures_from_validation_json(j):
    # j may be checkpoint-style (run_results) or single validation_result with "results"
    failures = []
    # If this is checkpoint full run with run_results
    if isinstance(j, dict) and "run_results" in j:
        for run_k, run_v in j["run_results"].items():
            # each run_v may contain 'validation_result'
            vr = run_v.get("validation_result") or run_v.get("validation_result_identifier") or run_v
            if isinstance(vr, dict) and "results" in vr:
                failures += extract_failures_from_validation_result(vr)
    # If it's a direct validation_result or a list
    elif isinstance(j, dict) and "results" in j:
        failures += extract_failures_from_validation_result(j)
    else:
        # try to find nested validation_result keys
        for k,v in j.items():
            if isinstance(v, dict) and "validation_result" in v:
                failures += extract_failures_from_validation_result(v["validation_result"])
            elif isinstance(v, dict) and "results" in v:
                failures += extract_failures_from_validation_result(v)
    return failures

def extract_failures_from_validation_result(vr):
    out = []
    suite = vr.get("meta", {}).get("expectation_suite_name") or "<unknown_suite>"
    batch_id = vr.get("meta", {}).get("batch_kwargs", {}).get("path") or vr.get("meta", {}).get("batch_kwargs", {}).get("datasource") or vr.get("meta", {}).get("batch_kwargs", {}) or "<batch?>"
    for r in vr.get("results", []):
        if not r.get("success", True):
            ec = r.get("expectation_config", {})
            etype = ec.get("expectation_type") or "<type?>"
            kwargs = ec.get("kwargs", {})
            column = kwargs.get("column") or kwargs.get("column_name") or "<no-column>"
            result = r.get("result", {})
            observed_value = result.get("observed_value", result.get("unexpected_count", "<obs?>"))
            unexpected_sample = result.get("partial_unexpected_list") or result.get("unexpected_list") or []
            out.append({
                "suite": suite,
                "batch": batch_id,
                "expectation_type": etype,
                "column": column,
                "kwargs": kwargs,
                "observed_value": observed_value,
                "unexpected_sample": unexpected_sample if isinstance(unexpected_sample, list) else [unexpected_sample],
            })
    return out

def print_failures(failures):
    for i, f in enumerate(failures, 1):
        print(f"{i}. suite: {f['suite']}")
        print(f"   batch: {f['batch']}")
        print(f"   expectation: {f['expectation_type']}")
        print(f"   column: {f['column']}")
        print(f"   kwargs: {json.dumps(f['kwargs'], default=str)}")
        print(f"   observed_value: {f['observed_value']}")
        if f['unexpected_sample']:
            sample = f['unexpected_sample'][:5]
            print(f"   unexpected sample (up to 5): {sample}")
        print("")
    print("----\n")

def run_and_inspect_checkpoint(context, checkpoint_name="validate_all_checkpoint"):
    print(f"Running checkpoint: {checkpoint_name} (programmatic run) ...\n")
    try:
        run_output = context.run_checkpoint(checkpoint_name=checkpoint_name)
    except Exception as e:
        print("Programmatic checkpoint run failed with exception:", e)
        return None
    # Save run output
    outdir = os.path.join(context.root_directory, "great_expectations", "uncommitted", "validations")
    if not os.path.isdir(outdir):
        outdir = os.path.join(context.root_directory, "uncommitted", "validations")
        os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outfile = os.path.join(outdir, f"auto_run_output_{ts}.json")
    try:
        with open(outfile, "w", encoding="utf-8") as fh:
            json.dump(run_output, fh, default=str, indent=2)
        print(f"Saved programmatic checkpoint run output to: {outfile}\n")
    except Exception as e:
        print("Warning: could not save run output:", e)
    # Inspect run_results structure for failures
    failures = []
    if isinstance(run_output, dict) and "run_results" in run_output:
        for run_id, run_item in run_output["run_results"].items():
            # run_item can contain 'validation_result'
            vr = run_item.get("validation_result") or run_item.get("validation_result_identifier") or run_item
            failures += extract_failures_from_validation_result(vr) if isinstance(vr, dict) else []
    else:
        # fallback: maybe it's a single validation_result
        failures += extract_failures_from_validation_json(run_output)
    if failures:
        print(f"Found {len(failures)} failure(s) in checkpoint run_results:\n")
        print_failures(failures)
    else:
        print("No failures found in programmatic checkpoint run_results.\n")
    return run_output

def main():
    context = DataContext()
    print("\nSTEP 1 — Scan saved validation JSON files")
    scan_saved_validations(context)
    print("\nSTEP 2 — Programmatically run checkpoint and inspect run_results")
    run_and_inspect_checkpoint(context, checkpoint_name="validate_all_checkpoint")
    print("STEP 3 — DONE\n")

if __name__ == "__main__":
    main()
