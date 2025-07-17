import sys
from importlib.metadata import distributions


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
    packages_list = [  # default packages
        "transformers",
        "accelerate",
        "torch",
        "tensorflow",
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
        "sklearn",
        "lightgbm",
        "vtk",
        "opencv-python",
        "keras",
        "matplotlib",
        "pydicom",
    ]
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
    get_python_packages()
