"""
Hugging Face download utility module

Provides functions to download models and datasets from the Hugging Face Hub.

Main functions:
- download_dataset: Download entire dataset repo using snapshot_download
- download_model: Download model repo using snapshot_download

For high-level dataset management (with registered datasets), use:
    from rdagent.scenarios.finetune.datasets import prepare, prepare_all

Environment variable configuration:
- HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN: Hugging Face access token
- FT_FILE_PATH: Root directory for finetuning files (managed by FT_RD_SETTING)

Usage example:
    from rdagent.scenarios.finetune.download.hf import download_dataset, download_model

    # Download dataset
    dataset_path = download_dataset("OpenMol/ChemCoTDataset", "/path/to/chemcot")

    # Download model
    model_path = download_model("Qwen/Qwen2.5-7B")
"""
