import sys
import json
from pathlib import Path


def pretty_float(val):
    try:
        import numpy as np
        if isinstance(val, float) or (hasattr(np, 'floating') and isinstance(val, np.floating)):
            return f"{float(val):.4f}"
    except Exception:
        pass
    return str(val)

def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_single_iteration.py <iteration_json_file>")
        sys.exit(1)
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Print iteration info
    info = data.get('iteration_info', {})
    print_section("Iteration Info")
    for k, v in info.items():
        print(f"{k:20}: {v}")

    # Print experiment info
    exp = data.get('experiment_info', {})
    print_section("Experiment Info")
    for k, v in exp.items():
        if k == 'config':
            continue
        print(f"{k:20}: {v}")

    # Print result data
    result = data.get('result_data', {})
    print_section("Result Data")
    for k in ['dataset', 'model', 'explanation_method', 'validation_status', 'used_tuned_params']:
        if k in result:
            print(f"{k:20}: {result[k]}")

    # Print model performance if available
    if 'model_performance' in result:
        print_section("Model Performance")
        perf = result['model_performance']
        if isinstance(perf, dict):
            for k, v in perf.items():
                print(f"{k:20}: {pretty_float(v)}")
        else:
            print(perf)

    # Print explanation evaluation results
    if 'evaluation_results' in result:
        print_section("Explanation Evaluation Results")
        evals = result['evaluation_results']
        if isinstance(evals, dict):
            for k, v in evals.items():
                print(f"{k:20}: {pretty_float(v)}")
        else:
            print(evals)

    # Print explanation results (optional, summary only)
    if 'explanation_results' in result:
        print_section("Explanation Results (summary)")
        expl = result['explanation_results']
        if isinstance(expl, dict):
            for k in ['generation_time', 'info']:
                if k in expl:
                    print(f"{k:20}: {expl[k]}")
        else:
            print(str(expl)[:300])

if __name__ == "__main__":
    main()
