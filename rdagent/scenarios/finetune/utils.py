def dataset_to_safe_component(dataset: str) -> str:
    """Make dataset string safe for a single path component.

    Replace path separators ("/" and "\\") with "__" to avoid nested dirs.
    Keep other characters unchanged to remain readable.
    """
    return dataset.replace("/", "__").replace("\\", "__")


def prev_model_dirname(model: str, dataset: str) -> str:
    """Generate prev_model directory name using model and dataset safely."""
    return f"{model}_{dataset_to_safe_component(dataset)}"
