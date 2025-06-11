import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.proposal.exp_gen.rag_hybrid_2 import HybridRAGSystem

from dotenv import load_dotenv


load_dotenv() 
# 初始化（会自动加载缓存）
api_backend = APIBackend()
rag_system = HybridRAGSystem(api_backend, cache_dir="./hypothesis_rag_cache")

# 定义问题
problems = {
  "Patient-level leakage from random patch sampling": {
    "problem": "Because there are only eight training whole-slide images, splitting the data at the patch level places tiles from the same kidney section into both the training and validation folds, allowing the model to memorize slide-specific texture instead of generalizing to unseen patients.",
    "reason": "The mean Dice on the leaderboard is computed per image; if validation leakage is not removed the offline score is over-estimated, leading to models that fail on the hidden test slides.",
    "label": "SCENARIO_PROBLEM"
  },
  "Extreme foreground-background pixel imbalance": {
    "problem": "Glomeruli occupy well under 1 % of the pixels, so standard cross-entropy or Dice losses converge to predicting empty masks that achieve deceptively high loss values yet score zero Dice when glomeruli are present.",
    "reason": "Addressing the imbalance with focal/Tversky losses or hard-positive mining directly increases the overlap term |X∩Y|, thereby improving the competition metric.",
    "label": "SCENARIO_PROBLEM"
  },
  "Resolution loss from global down-sampling of giga-pixel TIFFs": {
    "problem": "Uniformly shrinking whole-slide images to fit GPU memory makes many glomeruli smaller than a few pixels, rendering them impossible to segment and suppressing the numerator of the Dice score.",
    "reason": "Maintaining native resolution through smart tiling or multi-scale pyramids preserves object detail, producing more accurate masks and a higher Dice coefficient.",
    "label": "SCENARIO_PROBLEM"
  },
  "Prediction discontinuities at tile borders": {
    "problem": "Independent inference on non-overlapping tiles creates artificial seams—glomeruli cut at edges are either missed or double-predicted—reducing the intersection area after stitching.",
    "reason": "Using overlapping tiles with blending or padding aligns partial objects, boosting |X∩Y| and thus the overall Dice metric.",
    "label": "SCENARIO_PROBLEM"
  },
  "Uncorrected PAS stain variability across slides": {
    "problem": "Differences in staining intensity and color between patients shift the pixel distribution, causing the model to confuse background tissue with glomeruli on unseen slides.",
    "reason": "Applying stain normalization or color-constancy augmentations aligns data distributions across train and test sets, improving generalization and Dice performance.",
    "label": "SCENARIO_PROBLEM"
  },
  "Unsupported 'region' argument in tifffile": {
    "problem": "The DataLoader uses TiffPage.asarray(region=...) to crop sub‑windows, but tifffile’s API does not accept a 'region' keyword, so every worker raises TypeError and training never starts.",
    "reason": "Fixing the call (e.g. load the full page or use tifffile.TiffFile.asarray(...) followed by numpy slicing) will let the model actually receive image data, enabling training/inference and thus any Dice optimisation at all.",
    "label": "FEEDBACK_PROBLEM"
  },
  "Hard‑coded /kaggle/input path": {
    "problem": "find_data_folder() aborts with RuntimeError if the dataset is not located strictly under /kaggle/input/, making the script unusable on the competition server where the path differs.",
    "reason": "Replacing the rigid check with an environment‑agnostic search or CLI argument will allow the pipeline to locate images, generate predictions and consequently improve the Dice score instead of returning empty outputs.",
    "label": "FEEDBACK_PROBLEM"
  },
  "Mandatory openslide import without fallback": {
    "problem": "The pipeline exits early when openslide (or libopenslide) is missing, even though tifffile can read the provided .tiff slides; it writes an empty submission and terminates.",
    "reason": "Adding a tifffile‑based fallback for patch extraction removes the hard dependency, allowing full training/inference and therefore a chance to raise the Dice metric.",
    "label": "FEEDBACK_PROBLEM"
  },
  "Shape mismatch in Macenko stain normalisation": {
    "problem": "macenko_normalise() reshapes its output to (3, h, 3) instead of (h, w, 3) causing ValueError during augmentation, which stops training after the first fold.",
    "reason": "Correctly reconstructing the (h, w, 3) RGB array will keep the training loop running across all slides, letting colour‑normalised data reach the network and potentially improve Dice.",
    "label": "FEEDBACK_PROBLEM"
  },
  "Excessive full‑slide validation per epoch": {
    "problem": "Even after reducing epochs, the script still runs exhaustive full‑slide inference on every validation set at each epoch, dominating wall‑time and repeatedly causing 600 s timeouts.",
    "reason": "Limiting validation to a small patch subset or running it only after final epoch will keep the job within the 1‑hour limit, so the model can finish training and produce a submission that can be optimised for higher Dice.",
    "label": "FEEDBACK_PROBLEM"
  }
}

# 生成假设
results = rag_system.hypothesis_draft(
    no_sota_idea_path=None,  
    component_desc="Essay scoring optimization",
    scenario_desc="Improve model efficiency",
    exp_and_feedback_list_desc="Previous optimizations didn't help much",
    problems=problems
)

