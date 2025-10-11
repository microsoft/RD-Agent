"""
Hugging Face download utility module

Provides convenient functions to download models and datasets from the Hugging Face Hub.
Uses FT_RD_SETTING for unified path management in finetune scenarios.

Main functions:
- download_dataset: Download datasets
- download_model: Download models

Environment variable configuration:
- HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN: Hugging Face access token
- FT_FILE_PATH: Root directory for finetuning files (managed by FT_RD_SETTING)

Usage example:
    from rdagent.scenarios.finetune.download import download_dataset, download_model

    # Download dataset (uses FT_RD_SETTING.dataset_path by default)
    ds_path = download_dataset("shibing624/alpaca-zh", force=True)

    # Download model to specified directory (overrides default path)
    model_path = download_model("Qwen/Qwen2.5-7B", out_dir_root="/path/to/models")

    # Download model using default path (FT_RD_SETTING.model_path)
    model_path = download_model("Qwen/Qwen2.5-7B")

    # Download private model with token
    model_path = download_model("private/model", token="hf_xxx")

    # Download specific revision
    model_path = download_model("model/repo", revision="main")
"""
