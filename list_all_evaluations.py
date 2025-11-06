import json
from collections import defaultdict

# Load benchmark results
with open('results/experiment_20251020_163046/benchmark_results.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("COMPREHENSIVE LIST OF ALL EVALUATIONS")
print("="*80)

# Get all unique evaluation metrics
all_metrics = set()
for entry in data['comprehensive_results']:
    all_metrics.update(entry['evaluations'].keys())

print(f"\nTotal entries: {len(data['comprehensive_results'])}")
print(f"Total unique evaluation metrics: {len(all_metrics)}")

print("\n" + "="*80)
print("ALL EVALUATION METRICS")
print("="*80)

# Categorize metrics
basic_metrics = [
    'faithfulness',
    'stability',
    'monotonicity',
    'completeness',
    'simplicity',
    'sparsity',
    'consistency',
    'time_complexity'
]

advanced_metrics = [m for m in all_metrics if m.startswith('advanced_')]

print("\n1. BASIC EVALUATION METRICS:")
print("-" * 40)
for metric in sorted(basic_metrics):
    if metric in all_metrics:
        print(f"   - {metric}")

print("\n2. ADVANCED EVALUATION METRICS:")
print("-" * 40)
for metric in sorted(advanced_metrics):
    print(f"   - {metric.replace('advanced_', '')}")

# Count how many evaluations have each metric
print("\n" + "="*80)
print("METRIC COVERAGE")
print("="*80)

metric_counts = defaultdict(int)
for entry in data['comprehensive_results']:
    for metric in entry['evaluations'].keys():
        if entry['evaluations'][metric] is not None:
            metric_counts[metric] += 1

print(f"\nMetrics present in all {len(data['comprehensive_results'])} entries:")
print("-" * 40)
for metric in sorted(metric_counts.keys()):
    count = metric_counts[metric]
    percentage = (count / len(data['comprehensive_results'])) * 100
    print(f"   {metric:<50} {count:>4} ({percentage:>6.2f}%)")

# Show breakdown by modality
print("\n" + "="*80)
print("BREAKDOWN BY MODALITY")
print("="*80)

modality_breakdown = defaultdict(lambda: defaultdict(list))
for entry in data['comprehensive_results']:
    modality = entry['dataset_type']
    for metric in entry['evaluations'].keys():
        if entry['evaluations'][metric] is not None:
            modality_breakdown[modality][metric].append(entry)

for modality in sorted(modality_breakdown.keys()):
    print(f"\n{modality}:")
    print("-" * 40)
    for metric in sorted(modality_breakdown[modality].keys()):
        count = len(modality_breakdown[modality][metric])
        print(f"   {metric:<50} {count:>4} entries")

# Show breakdown by explanation method
print("\n" + "="*80)
print("BREAKDOWN BY EXPLANATION METHOD")
print("="*80)

method_breakdown = defaultdict(lambda: defaultdict(list))
for entry in data['comprehensive_results']:
    method = entry['explanation_method']
    for metric in entry['evaluations'].keys():
        if entry['evaluations'][metric] is not None:
            method_breakdown[method][metric].append(entry)

for method in sorted(method_breakdown.keys()):
    print(f"\n{method}:")
    print("-" * 40)
    metrics = method_breakdown[method]
    print(f"   Total evaluations: {len(metrics)} different metrics")
    print(f"   Metrics: {', '.join(sorted(metrics.keys())[:5])}...")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Total datasets: {len(set(e['dataset'] for e in data['comprehensive_results']))}
Total models: {len(set(e['model'] for e in data['comprehensive_results']))}
Total explanation methods: {len(set(e['explanation_method'] for e in data['comprehensive_results']))}
Total modalities: {len(set(e['dataset_type'] for e in data['comprehensive_results']))}
Total entries: {len(data['comprehensive_results'])}
Total unique metrics: {len(all_metrics)}

Basic metrics: {len([m for m in all_metrics if not m.startswith('advanced_')])}
Advanced metrics: {len(advanced_metrics)}
""")
