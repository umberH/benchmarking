#!/usr/bin/env python3
"""
Integration script to add run organization to existing benchmark system
"""

from pathlib import Path
from run_organization_demo import create_run_structure

def integrate_with_main_py():
    """Show how to integrate run organization with main.py"""
    
    print("🔧 Integration with Main Benchmarking System")
    print("=" * 50)
    
    # Example integration code for main.py
    integration_code = '''
# Add to main.py before creating XAIBenchmark instance:

def setup_run_organization(args):
    """Setup run-based organization"""
    from run_organization_demo import create_run_structure
    
    # Determine run type based on arguments
    if args.comprehensive:
        run_type = "comprehensive"
    elif args.iterative:
        run_type = "iterative"
    else:
        run_type = "standard"
    
    # Create organized run structure
    run_info = create_run_structure(run_type=run_type)
    
    print(f"📁 Created run: {run_info['run_metadata']['run_name']}")
    print(f"🔗 Latest run link: results/runs/latest")
    
    return run_info['paths']['run_dir']

# In main() function, replace:
# benchmark = XAIBenchmark(config, Path("results"))

# With:
# run_dir = setup_run_organization(args)
# benchmark = XAIBenchmark(config, run_dir)
'''
    
    print("📝 Integration Code:")
    print(integration_code)
    
    # Show benefits
    print("\n🎯 Benefits for Users:")
    print("   ✅ Each run is completely isolated")
    print("   ✅ Easy to compare results across different experiments")
    print("   ✅ No overwriting of previous results")
    print("   ✅ Clean organization of models, logs, and reports")
    print("   ✅ Dashboard can now show multiple runs")
    print("   ✅ Complete audit trail of all experiments")
    
    # Show usage examples
    print("\n💡 Usage Examples:")
    print("   python main.py --comprehensive")
    print("   → Creates: results/runs/run_20250818_144500_comprehensive/")
    print("   ")
    print("   python main.py --iterative adult_income decision_tree shap")
    print("   → Creates: results/runs/run_20250818_144600_iterative/")
    print("   ")
    print("   python main.py")
    print("   → Creates: results/runs/run_20250818_144700_standard/")

def show_dashboard_integration():
    """Show how dashboard will work with run organization"""
    
    print("\n🖥️ Dashboard Integration Benefits")
    print("=" * 40)
    
    print("📊 Enhanced Dashboard Features:")
    print("   • Select specific runs from dropdown")
    print("   • Compare performance across multiple runs")
    print("   • View run metadata and experiment details")
    print("   • Track experiment evolution over time")
    print("   • Load run-specific detailed explanations")
    
    print("\n🔄 Multi-Run Comparison:")
    print("   • Side-by-side performance analysis")
    print("   • Track method improvements over time")
    print("   • Identify best-performing configurations")
    print("   • Analyze consistency across runs")

def create_sample_runs():
    """Create sample runs to demonstrate the system"""
    
    print("\n🧪 Creating Sample Runs for Testing")
    print("=" * 40)
    
    run_types = ["comprehensive", "targeted", "experimental"]
    
    for run_type in run_types:
        run_info = create_run_structure(run_type=run_type)
        print(f"✅ Created sample {run_type} run: {run_info['run_metadata']['run_name']}")
    
    print(f"\n📁 All runs stored in: results/runs/")
    print(f"🔗 Latest run: results/runs/latest")

if __name__ == "__main__":
    integrate_with_main_py()
    show_dashboard_integration()
    
    # Ask user if they want to create sample runs
    response = input("\n❓ Create sample runs for testing? (y/n): ").lower().strip()
    if response == 'y':
        create_sample_runs()
        print("\n🎉 Sample runs created! You can now test the dashboard with multiple runs.")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Integrate run organization into main.py")
    print(f"   2. Test comprehensive benchmarking with new structure")
    print(f"   3. Use dashboard to compare multiple runs")
    print(f"   4. Enjoy organized, professional experiment management!")