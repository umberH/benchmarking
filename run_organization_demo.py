#!/usr/bin/env python3
"""
Demo script showing how to implement run-based folder organization
"""

import os
import json
from pathlib import Path
from datetime import datetime
import shutil

def create_run_structure(base_results_dir="results", run_type="comprehensive"):
    """
    Create organized run-based folder structure
    
    Args:
        base_results_dir: Base results directory
        run_type: Type of run (comprehensive, targeted, etc.)
    
    Returns:
        Dictionary with run information and paths
    """
    
    # Generate run metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{run_type}"
    run_name = f"run_{run_id}"
    
    # Create directory structure
    base_path = Path(base_results_dir)
    runs_dir = base_path / "runs"
    run_dir = runs_dir / run_name
    
    # Create all subdirectories
    subdirectories = {
        'run_dir': run_dir,
        'iterations': run_dir / "iterations",
        'detailed_explanations': run_dir / "detailed_explanations",
        'models': run_dir / "models", 
        'logs': run_dir / "logs",
        'reports': run_dir / "reports",
        'config': run_dir / "config"
    }
    
    # Create directories
    for dir_path in subdirectories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to latest run
    latest_link = runs_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # On Windows, use junction instead of symlink if needed
        if os.name == 'nt':
            import subprocess
            subprocess.run(['mklink', '/J', str(latest_link), str(run_dir)], 
                         shell=True, check=False)
        else:
            latest_link.symlink_to(run_name, target_is_directory=True)
    except Exception as e:
        print(f"Warning: Could not create 'latest' link: {e}")
    
    # Create run metadata file
    run_metadata = {
        'run_id': run_id,
        'run_name': run_name,
        'timestamp': timestamp,
        'run_type': run_type,
        'created_at': datetime.now().isoformat(),
        'status': 'initialized',
        'paths': {k: str(v) for k, v in subdirectories.items()}
    }
    
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    # Copy current results to new structure (if they exist)
    migrate_existing_results(base_path, run_dir)
    
    return {
        'run_metadata': run_metadata,
        'paths': subdirectories
    }

def migrate_existing_results(old_results_dir: Path, new_run_dir: Path):
    """Migrate existing results to new run structure"""
    
    migration_log = []
    
    # Files to migrate
    files_to_migrate = [
        ('benchmark_results.json', 'benchmark_results.json'),
        ('comprehensive_report.md', 'reports/comprehensive_report.md'),
        ('detailed_results.json', 'detailed_results.json')
    ]
    
    for old_file, new_path in files_to_migrate:
        old_path = old_results_dir / old_file
        new_file_path = new_run_dir / new_path
        
        if old_path.exists():
            # Ensure parent directory exists
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy file
            shutil.copy2(old_path, new_file_path)
            migration_log.append(f"Migrated: {old_file} -> {new_path}")
    
    # Migrate directories
    dirs_to_migrate = [
        ('iterations', 'iterations'),
        ('detailed_explanations', 'detailed_explanations')
    ]
    
    for old_dir, new_dir in dirs_to_migrate:
        old_dir_path = old_results_dir / old_dir
        new_dir_path = new_run_dir / new_dir
        
        if old_dir_path.exists():
            # Copy directory contents
            if old_dir_path.is_dir():
                for item in old_dir_path.iterdir():
                    dest = new_dir_path / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                migration_log.append(f"Migrated directory: {old_dir} -> {new_dir}")
    
    # Save migration log
    if migration_log:
        log_file = new_run_dir / "migration_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"Migration performed at: {datetime.now().isoformat()}\n")
            f.write("Files migrated:\n")
            for entry in migration_log:
                f.write(f"  - {entry}\n")

def list_available_runs(base_results_dir="results"):
    """List all available runs"""
    
    runs_dir = Path(base_results_dir) / "runs"
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            metadata_file = run_dir / "run_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    runs.append(metadata)
                except Exception as e:
                    # Create minimal metadata for runs without metadata file
                    runs.append({
                        'run_name': run_dir.name,
                        'run_id': run_dir.name.replace('run_', ''),
                        'timestamp': 'unknown',
                        'run_type': 'unknown',
                        'paths': {'run_dir': str(run_dir)}
                    })
    
    return sorted(runs, key=lambda x: x.get('timestamp', ''), reverse=True)

def demo_run_organization():
    """Demonstrate the run organization system"""
    
    print("🏗️ XAI Benchmarking Run Organization Demo")
    print("=" * 50)
    
    # Create a new run
    print("1. Creating new comprehensive run...")
    run_info = create_run_structure(run_type="comprehensive")
    
    print(f"✅ Created run: {run_info['run_metadata']['run_name']}")
    print(f"📁 Run directory: {run_info['paths']['run_dir']}")
    
    # Show directory structure
    print(f"\n📂 Directory structure:")
    for name, path in run_info['paths'].items():
        print(f"   {name:20}: {path}")
    
    # Create another run to demonstrate multiple runs
    print(f"\n2. Creating targeted run...")
    run_info2 = create_run_structure(run_type="targeted")
    
    print(f"✅ Created run: {run_info2['run_metadata']['run_name']}")
    
    # List available runs
    print(f"\n3. Available runs:")
    available_runs = list_available_runs()
    for i, run in enumerate(available_runs, 1):
        print(f"   {i}. {run['run_name']} ({run['run_type']}) - {run['timestamp']}")
    
    # Show benefits
    print(f"\n🎯 Benefits of Run Organization:")
    print(f"   ✅ Clean separation of different experiments")
    print(f"   ✅ Easy to compare results across runs")
    print(f"   ✅ Automatic migration of existing results")
    print(f"   ✅ Latest run always accessible via 'latest' link")
    print(f"   ✅ Complete audit trail with metadata")
    print(f"   ✅ Organized storage for models, logs, and reports")
    
    # Show usage in dashboard
    print(f"\n🖥️ Dashboard Integration:")
    print(f"   • Auto-discover all runs in results/runs/")
    print(f"   • Compare performance across different runs")
    print(f"   • Load specific run data for analysis")
    print(f"   • Track experiment history and evolution")

if __name__ == "__main__":
    demo_run_organization()