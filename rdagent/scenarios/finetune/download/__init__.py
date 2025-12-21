"""
Hugging Face download utility module

Provides convenient functions to download models and load datasets from the Hugging Face Hub.
Uses FT_RD_SETTING for unified path management in finetune scenarios.

Main functions:
- load_dataset_split: Load a specific split from a HF dataset as Dataset object
- export_dataset: Export Dataset object to local file (json, jsonl, csv, parquet)
- download_model: Download models using snapshot_download

For high-level dataset management (with registered datasets), use:
    from rdagent.scenarios.finetune.datasets import prepare, load_split

Environment variable configuration:
- HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN: Hugging Face access token
- FT_FILE_PATH: Root directory for finetuning files (managed by FT_RD_SETTING)

Usage example:
    from rdagent.scenarios.finetune.download.hf import load_dataset_split, export_dataset, download_model

    # Load specific split from dataset (does not download entire repo)
    ds = load_dataset_split("LG-AI-Research/PANORAMA", split="train", data_dir="PAR4PC")

    # Export to local file
    export_dataset(ds, "/path/to/train.json", format="json")

    # Download model
    model_path = download_model("Qwen/Qwen2.5-7B")
"""
