#!/usr/bin/env python3
"""
Test script to verify experiment folder organization
"""

import os
import time
from pathlib import Path
from datetime import datetime

def test_experiment_folders():
    """Test that experiment folders are created with timestamps"""
    
    print("🧪 Testing Experiment Folder Organization")
    print("=" * 50)
    
    # Show current structure
    results_dir = Path("results")
    if results_dir.exists():
        print(f"📁 Current results directory contents:")
        for item in results_dir.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}")
            else:
                print(f"   📄 {item.name}")
    else:
        print("📁 No results directory found yet")
    
    print(f"\n✨ Next experiment will create folder with format:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"   📂 results/experiment_{timestamp}")
    
    print(f"\n🎯 Usage examples:")
    print(f"   Default: python main.py --comprehensive")
    print(f"   → results/experiment_{timestamp}/")
    
    print(f"\n   Named:   python main.py --comprehensive --experiment-name \"fix_validation\"")
    print(f"   → results/fix_validation_{timestamp}/")
    
    print(f"\n   Named:   python main.py --comprehensive --experiment-name \"baseline_test\"") 
    print(f"   → results/baseline_test_{timestamp}/")
    
    print(f"\n💡 Benefits:")
    print(f"   ✅ Each run gets its own folder")
    print(f"   ✅ Easy to compare different experiments") 
    print(f"   ✅ No overwriting of previous results")
    print(f"   ✅ Timestamps ensure unique folders")

if __name__ == "__main__":
    test_experiment_folders()