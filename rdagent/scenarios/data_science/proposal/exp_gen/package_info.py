import sys
from importlib.metadata import distributions

# Kaggle competition packages - based on usage frequency
PYTHON_BASE_PACKAGES = ["catboost", "lightgbm", "numpy", "optuna", "pandas", "scikit-learn", "scipy", "shap", "xgboost"]

PYTHON_ADVANCED_PACKAGES = [
    "accelerate",
    "albumentations",
    "bayesian-optimization",
    "category_encoders",
    "datasets",
    "featuretools",
    "imbalanced-learn",
    "nltk",
    "opencv-python",
    "pillow",
    "polars",
    "sentence-transformers",
    "spacy",
    "tensorflow",
    "timm",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
]


def get_all_excepted_packages():
    """Get flattened list of all packages"""
    all_packages = PYTHON_BASE_PACKAGES + PYTHON_ADVANCED_PACKAGES
    return sorted(set(all_packages))


def get_available_recommended_packages_prompt():
    """Generate prompt template for dynamically detected available packages"""
    installed_packages = get_installed_packages()

    # Check which packages are actually installed
    base_available = [pkg for pkg in PYTHON_BASE_PACKAGES if pkg.lower() in installed_packages]
    advanced_available = [pkg for pkg in PYTHON_ADVANCED_PACKAGES if pkg.lower() in installed_packages]

    # Build prompt
    prompt_parts = ["# Available packages in environment:\n"]

    if base_available:
        prompt_parts.append("## [Basic Libraries] (general tools for data science tasks):")
        prompt_parts.append(f"- {', '.join(base_available)}")
        prompt_parts.append("")

    if advanced_available:
        prompt_parts.append("## [Advanced Tools] (specialized for specific domains):")
        prompt_parts.append(f"- {', '.join(advanced_available)}")
        prompt_parts.append("")

    prompt_parts.append(
        "You should choose appropriate tool combinations based on the specific context and current situation. Feel free to use any other packages you think are necessary to achieve the best performance."
    )

    return "\n".join(prompt_parts).strip()


def print_available_packages_prompt():
    """Print the available packages prompt to stdout for external consumption"""
    prompt = get_available_recommended_packages_prompt()
    print(prompt)


def get_installed_packages():
    return {dist.metadata["Name"].lower(): dist.version for dist in distributions()}


def print_filtered_packages(installed_packages, filtered_packages):
    to_print = []
    for package_name in filtered_packages:
        version = installed_packages.get(package_name.lower())
        if version:
            to_print.append((package_name, version))
    if not to_print:
        print("=== No matching packages found ===")
    else:
        print("=== Installed Packages ===")
        for package_name, version in to_print:
            # Print package name and version in the format "package_name==version"
            print(f"{package_name}=={version}")


def get_python_packages():
    # Allow the caller to pass a custom package list via command-line arguments.
    # Example: `python package_info.py pandas torch scikit-learn`
    # If no extra arguments are provided we fall back to the original default list
    # to keep full backward-compatibility.
    packages_list = get_all_excepted_packages()
    if len(sys.argv) > 1:
        packages_list = list(set(packages_list) | set(sys.argv[1:]))

    installed_packages = get_installed_packages()

    print_filtered_packages(installed_packages, packages_list)

    # TODO: Handle missing packages.
    # Report packages that are requested by the LLM but are not installed.
    missing_pkgs = [pkg for pkg in packages_list if pkg.lower() not in installed_packages]
    if missing_pkgs:
        print("\n=== Missing Packages (Avoid using these packages) ===")
        for pkg in missing_pkgs:
            print(pkg)


if __name__ == "__main__":
    # Check if we should print available packages prompt
    if len(sys.argv) > 1 and sys.argv[1] == "--packages-prompt":
        print_available_packages_prompt()
    else:
        get_python_packages()
