"""
Check which datasets from your config are available on OpenML
and provide OpenML dataset IDs for easy access
"""

# OpenML dataset mapping for your configured datasets
OPENML_DATASET_MAPPING = {
    # Tabular datasets
    "adult_income": {
        "openml_id": 1590,  # Adult (a9a)
        "name": "adult",
        "description": "Adult Income prediction dataset",
        "task_type": "binary_classification",
        "available": True
    },
    "breast_cancer": {
        "openml_id": 13,    # breast-cancer
        "name": "breast-cancer",
        "description": "Breast Cancer Wisconsin dataset",
        "task_type": "binary_classification",
        "available": True
    },
    "heart_disease": {
        "openml_id": 4,     # cleveland
        "name": "cleveland",
        "description": "Heart Disease Cleveland dataset",
        "task_type": "binary_classification",
        "available": True
    },
    "german_credit": {
        "openml_id": 31,    # credit-g
        "name": "credit-g",
        "description": "German Credit Risk dataset",
        "task_type": "binary_classification",
        "available": True
    },
    "iris": {
        "openml_id": 61,    # iris
        "name": "iris",
        "description": "Iris flower classification",
        "task_type": "multiclass_classification",
        "available": True
    },
    "wine_classification": {
        "openml_id": 187,   # wine
        "name": "wine",
        "description": "Wine classification dataset",
        "task_type": "multiclass_classification",
        "available": True
    },
    "diabetes": {
        "openml_id": 37,    # diabetes
        "name": "diabetes",
        "description": "Pima Indians Diabetes (binary classification)",
        "task_type": "binary_classification",
        "available": True,
        "note": "This is binary classification, not 3-class as in your config"
    },

    # Datasets NOT directly available on OpenML (need alternatives)
    "compas": {
        "openml_id": None,
        "alternative_openml_id": 42890,  # Similar fairness dataset
        "name": "compas-two-years",
        "description": "COMPAS recidivism dataset - not on OpenML, but similar fairness datasets available",
        "task_type": "binary_classification",
        "available": False,
        "note": "Use ProPublica COMPAS data or similar fairness dataset from OpenML"
    },
    "wine_quality": {
        "openml_id": 40691,  # winequality-red
        "name": "winequality-red",
        "description": "Wine Quality dataset (can be converted to 3-class)",
        "task_type": "regression_convertible_to_multiclass",
        "available": True,
        "note": "Originally regression, needs binning for 3-class"
    },
    "digits": {
        "openml_id": 554,   # mnist_784 (similar)
        "name": "mnist_784",
        "description": "MNIST digits (28x28 flattened to 784 features)",
        "task_type": "multiclass_classification",
        "available": True,
        "note": "MNIST available, sklearn digits (8x8) not directly on OpenML"
    }
}

# Additional OpenML datasets that would work well with your framework
RECOMMENDED_ADDITIONAL_DATASETS = {
    "titanic": {
        "openml_id": 40945,
        "name": "titanic",
        "description": "Titanic survival prediction",
        "task_type": "binary_classification"
    },
    "bank_marketing": {
        "openml_id": 1461,
        "name": "bank-marketing",
        "description": "Bank marketing dataset",
        "task_type": "binary_classification"
    },
    "mushroom": {
        "openml_id": 24,
        "name": "mushroom",
        "description": "Mushroom classification (safe/poisonous)",
        "task_type": "binary_classification"
    },
    "car_evaluation": {
        "openml_id": 21,
        "name": "car",
        "description": "Car evaluation dataset",
        "task_type": "multiclass_classification"
    },
    "segment": {
        "openml_id": 40984,
        "name": "segment",
        "description": "Image segmentation dataset",
        "task_type": "multiclass_classification"
    }
}


def print_openml_dataset_info():
    """Print information about OpenML dataset availability"""

    print("🗃️  OPENML DATASET AVAILABILITY FOR YOUR CONFIG")
    print("=" * 70)

    print("\n✅ AVAILABLE ON OPENML:")
    available_count = 0
    for dataset_name, info in OPENML_DATASET_MAPPING.items():
        if info["available"]:
            available_count += 1
            print(f"   {dataset_name:20} → OpenML ID: {info['openml_id']:5} ({info['name']})")
            if "note" in info:
                print(f"      📝 Note: {info['note']}")

    print(f"\n❌ NOT DIRECTLY AVAILABLE:")
    for dataset_name, info in OPENML_DATASET_MAPPING.items():
        if not info["available"]:
            print(f"   {dataset_name:20} → {info['description']}")
            if "alternative_openml_id" in info:
                print(f"      🔄 Alternative: OpenML ID {info['alternative_openml_id']} ({info['name']})")
            if "note" in info:
                print(f"      📝 Note: {info['note']}")

    print(f"\n📊 SUMMARY:")
    total = len(OPENML_DATASET_MAPPING)
    print(f"   Available on OpenML: {available_count}/{total} ({available_count/total:.1%})")

    print(f"\n🎯 RECOMMENDED ADDITIONAL OPENML DATASETS:")
    for dataset_name, info in RECOMMENDED_ADDITIONAL_DATASETS.items():
        print(f"   {dataset_name:20} → OpenML ID: {info['openml_id']:5} ({info['description']})")


def generate_openml_config():
    """Generate a config snippet for OpenML datasets"""

    print("\n" + "=" * 70)
    print("🔧 OPENML CONFIG SNIPPET")
    print("=" * 70)

    print("\n# Add this to your data configuration:")
    print("data:")
    print("  tabular_datasets:")

    for dataset_name, info in OPENML_DATASET_MAPPING.items():
        if info["available"] and info["task_type"] in ["binary_classification", "multiclass_classification"]:
            print(f"    - name: \"{dataset_name}\"")
            print(f"      source: \"openml\"")
            print(f"      openml_id: {info['openml_id']}")
            print(f"      description: \"{info['description']}\"")
            print(f"      task_type: \"{info['task_type']}\"")
            print(f"      mandatory: true")
            if "note" in info:
                print(f"      # Note: {info['note']}")
            print()


def generate_openml_loader_code():
    """Generate Python code to load datasets from OpenML"""

    print("\n" + "=" * 70)
    print("🐍 PYTHON CODE TO LOAD FROM OPENML")
    print("=" * 70)

    print("""
# Install openml if you haven't already:
# pip install openml

import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load any dataset from OpenML
def load_openml_dataset(dataset_id, target_column=None):
    \"\"\"Load dataset from OpenML\"\"\"
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=target_column
    )
    return X, y, dataset.name, dataset.description

# Load your configured datasets:""")

    for dataset_name, info in OPENML_DATASET_MAPPING.items():
        if info["available"]:
            print(f"""
# {dataset_name.upper()}
X_{dataset_name}, y_{dataset_name}, name, desc = load_openml_dataset({info['openml_id']})
print(f"Loaded {{name}}: {{X_{dataset_name}.shape}} samples, {{y_{dataset_name}.unique()}} classes")""")


def check_openml_installation():
    """Check if OpenML is installed and provide installation instructions"""

    print("\n" + "=" * 70)
    print("🔧 OPENML SETUP INSTRUCTIONS")
    print("=" * 70)

    try:
        import openml
        print("✅ OpenML is already installed!")
        print(f"   Version: {openml.__version__}")

        # Test connection
        try:
            dataset = openml.datasets.get_dataset(61)  # Iris
            print("✅ OpenML connection successful!")
            print(f"   Test dataset loaded: {dataset.name}")
        except Exception as e:
            print(f"❌ OpenML connection failed: {e}")
            print("   Check your internet connection")

    except ImportError:
        print("❌ OpenML not installed")
        print("\n📦 To install OpenML:")
        print("   pip install openml")
        print("   # or")
        print("   conda install -c conda-forge openml")

    print(f"\n🔑 OpenML API Key (optional but recommended):")
    print("   1. Create account at https://openml.org")
    print("   2. Get API key from your profile")
    print("   3. Set it: openml.config.apikey = 'YOUR_API_KEY'")
    print("   4. Or save to ~/.openml/config")


if __name__ == "__main__":
    print("🚀 OPENML DATASET CHECKER FOR XAI BENCHMARKING")

    # Check OpenML installation
    check_openml_installation()

    # Show dataset availability
    print_openml_dataset_info()

    # Generate config
    generate_openml_config()

    # Generate Python code
    generate_openml_loader_code()

    print(f"\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("1. Install OpenML: pip install openml")
    print("2. Update your data manager to support OpenML datasets")
    print("3. Use the OpenML IDs provided above")
    print("4. Consider adding the recommended datasets for more variety")
    print("5. Test with a few datasets first before running full benchmarks")

    print(f"\n💡 TIP: Start with these reliable OpenML datasets:")
    print("   - adult (ID: 1590) - Large, well-known")
    print("   - iris (ID: 61) - Small, fast for testing")
    print("   - breast-cancer (ID: 13) - Medical dataset")
    print("   - wine (ID: 187) - Multi-class")
    print("   - titanic (ID: 40945) - Popular benchmark")