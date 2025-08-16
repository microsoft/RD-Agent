from rdagent.scenarios.finetune.download.hf import download_dataset, download_model

__all__ = ["download_model", "download_dataset"]

"""
Hugging Face download utility module

Provides convenient functions to download models and datasets from the Hugging Face Hub.

Main functions:
- download_dataset: Download datasets
- download_model: Download models

Environment variable configuration:
- HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN: Hugging Face access token
- FT_FILE_PATH: Root directory for finetuning files (recommended)
- DS_LOCAL_DATA_PATH: Local data path for data science
- HF_MODEL_PATH / MODEL_DIR / MODELS_DIR: Model storage path

Usage example:
    from rdagent.scenarios.finetune.download import download_dataset, download_model

    # Download dataset (supports force overwrite)
    ds_path = download_dataset("shibing624/alpaca-zh", force=True)

    # Download model to specified directory
    model_path = download_model("Qwen/Qwen2.5-7B", out_dir_root="/path/to/models")

    # Download private model with token
    model_path = download_model("private/model", token="hf_xxx")

    # Download specific revision
    model_path = download_model("model/repo", revision="main")
"""
