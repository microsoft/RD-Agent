def safe_path_component(s: str) -> str:
    """Convert a string to a safe path component by replacing path separators with '__'."""
    return s.replace("/", "__").replace("\\", "__")


def prev_model_dirname(model: str, dataset: str) -> str:
    """Generate prev_model directory name using safe model and dataset names."""
    return f"{safe_path_component(model)}_{safe_path_component(dataset)}"
