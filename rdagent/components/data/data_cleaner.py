"""Data cleaner for SFT dataset preparation.

Performs data cleaning operations:
1. Deduplication (based on instruction+output hash)
2. Length filtering (min/max thresholds)
3. Quality scoring (LLM-based, with sampling strategy, parallel execution)
"""

import concurrent.futures
import hashlib
import json
import logging
from concurrent.futures import as_completed
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from rdagent.utils.agent.tpl import T

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and filter Alpaca-format datasets."""

    # Default configuration
    DEFAULT_CONFIG = {
        "min_instruction_len": 5,
        "min_output_len": 10,
        "max_instruction_len": 2000,
        "max_output_len": 4000,
        "quality_threshold": 7.0,
        "max_samples_for_scoring": 10000,
        "batch_size": 10,  # Number of samples to score in one LLM call
        "save_frequency": 10,  # Save intermediate results every N batches (0 = disable)
    }

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        """Initialize data cleaner.

        Args:
            llm_client: LLM client for quality scoring (optional)
            config: Configuration dict (uses defaults if not specified)
        """
        self.llm_client = llm_client
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def clean(
        self,
        alpaca_samples: List[Dict[str, str]],
        task_description: str = "",
        enable_quality_scoring: bool = False,
        max_workers: int = 20,
        save_callback: Any = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Clean Alpaca-format samples.

        Args:
            alpaca_samples: List of Alpaca samples
            task_description: User's task description (for quality scoring)
            enable_quality_scoring: Whether to enable LLM quality scoring
            max_workers: Number of parallel workers for quality scoring (default: 20)
            save_callback: Optional callback function(samples) to save intermediate results

        Returns:
            Tuple of (cleaned_samples, stats)
        """
        logger.info(f"Starting data cleaning on {len(alpaca_samples)} samples")

        stats = {"original_samples": len(alpaca_samples)}

        # Step 1: Deduplication
        samples = self._deduplicate(alpaca_samples)
        stats["after_dedup"] = len(samples)
        stats["dedup_removed"] = stats["original_samples"] - stats["after_dedup"]
        logger.info(
            f"After dedup: {stats['after_dedup']} samples "
            f"(removed {stats['dedup_removed']}, {stats['dedup_removed']/stats['original_samples']*100:.1f}%)"
        )

        # Step 2: Length filtering
        samples = self._filter_by_length(samples)
        stats["after_length_filter"] = len(samples)
        stats["length_filter_removed"] = stats["after_dedup"] - stats["after_length_filter"]
        logger.info(
            f"After length filter: {stats['after_length_filter']} samples "
            f"(removed {stats['length_filter_removed']}, {stats['length_filter_removed']/stats['after_dedup']*100:.1f}%)"
        )

        # Step 3: Quality scoring (optional, parallel)
        if enable_quality_scoring and self.llm_client:
            samples = self._filter_by_quality(samples, task_description, max_workers, save_callback)
            stats["after_quality_filter"] = len(samples)
            stats["quality_filter_removed"] = stats["after_length_filter"] - stats["after_quality_filter"]
            logger.info(
                f"After quality filter: {stats['after_quality_filter']} samples "
                f"(removed {stats['quality_filter_removed']}, "
                f"{stats['quality_filter_removed']/stats['after_length_filter']*100:.1f}%)"
            )
        else:
            stats["after_quality_filter"] = stats["after_length_filter"]
            stats["quality_filter_removed"] = 0

        stats["final_samples"] = len(samples)
        stats["total_removed"] = stats["original_samples"] - stats["final_samples"]
        stats["retention_rate"] = stats["final_samples"] / stats["original_samples"]

        logger.info(
            f"Cleaning complete: {stats['final_samples']}/{stats['original_samples']} samples retained "
            f"({stats['retention_rate']*100:.1f}%)"
        )

        return samples, stats

    def _deduplicate(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate samples based on instruction+output hash.

        Args:
            samples: List of Alpaca samples

        Returns:
            Deduplicated samples
        """
        seen_hashes = set()
        unique_samples = []

        for sample in samples:
            # Compute hash of instruction+output
            hash_key = self._compute_sample_hash(sample)

            if hash_key not in seen_hashes:
                seen_hashes.add(hash_key)
                unique_samples.append(sample)

        return unique_samples

    def _compute_sample_hash(self, sample: Dict[str, str]) -> str:
        """Compute hash of sample for deduplication.

        Args:
            sample: Alpaca sample

        Returns:
            Hash string
        """
        # Use instruction+output for hash (ignore input field)
        content = sample["instruction"] + sample["output"]
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _filter_by_length(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter samples by length constraints.

        Args:
            samples: List of Alpaca samples

        Returns:
            Filtered samples
        """
        filtered_samples = []

        for sample in samples:
            instruction_len = len(sample["instruction"])
            output_len = len(sample["output"])

            # Check min length
            if instruction_len < self.config["min_instruction_len"]:
                continue
            if output_len < self.config["min_output_len"]:
                continue

            # Check max length
            if instruction_len > self.config["max_instruction_len"]:
                continue
            if output_len > self.config["max_output_len"]:
                continue

            filtered_samples.append(sample)

        return filtered_samples

    def _filter_by_quality(
        self, samples: List[Dict[str, str]], task_description: str, max_workers: int = 20, save_callback: Any = None
    ) -> List[Dict[str, str]]:
        """Filter samples by LLM quality scoring (parallel execution).

        Uses sampling strategy to reduce cost:
        - If samples > max_samples_for_scoring, only score the first N samples
        - Scores samples in batches using parallel workers

        Args:
            samples: List of Alpaca samples
            task_description: User's task description
            max_workers: Number of parallel workers (default: 20)
            save_callback: Optional callback function(samples) to save intermediate results

        Returns:
            High-quality samples (score >= threshold)
        """
        max_samples = self.config["max_samples_for_scoring"]
        threshold = self.config["quality_threshold"]

        # Sampling strategy
        if len(samples) > max_samples:
            logger.info(f"Sampling {max_samples} out of {len(samples)} samples for quality scoring")
            samples_to_score = samples[:max_samples]
            samples_without_scoring = samples[max_samples:]
        else:
            samples_to_score = samples
            samples_without_scoring = []

        # Split into batches
        batch_size = self.config["batch_size"]
        batches = []
        for i in range(0, len(samples_to_score), batch_size):
            batch = samples_to_score[i : i + batch_size]
            batches.append((i // batch_size, batch))

        total_batches = len(batches)
        logger.info(f"Scoring {len(samples_to_score)} samples in {total_batches} batches with {max_workers} workers")

        # Parallel batch scoring
        high_quality_samples = []
        save_frequency = self.config.get("save_frequency", 0)
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scoring tasks
            future_to_batch = {}
            for batch_idx, batch in batches:
                future = executor.submit(self._score_batch_wrapper, batch, task_description, batch_idx, total_batches)
                future_to_batch[future] = (batch_idx, batch)

            # Collect results with progress bar
            with tqdm(total=total_batches, desc="✨ Quality scoring", unit="batch", ncols=100, colour="green") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx, batch = future_to_batch[future]
                    completed += 1

                    try:
                        scores = future.result()

                        # Filter by threshold
                        for sample, score_info in zip(batch, scores):
                            if score_info["score"] >= threshold:
                                high_quality_samples.append(sample)
                            else:
                                logger.debug(
                                    f"Filtered out sample (score={score_info['score']:.1f}): "
                                    f"{sample['instruction'][:50]}..."
                                )

                        # Save intermediate results if enabled
                        if save_callback and save_frequency > 0 and completed % save_frequency == 0:
                            save_callback(high_quality_samples)
                            logger.info(f"💾 Progress saved: {len(high_quality_samples)} high-quality samples")

                        # Update progress bar
                        pbar.set_postfix({"high_quality": len(high_quality_samples), "threshold": threshold})
                        pbar.update(1)

                    except Exception as e:
                        logger.warning(f"Failed to score batch {batch_idx}: {e}. Keeping all samples in batch.")
                        high_quality_samples.extend(batch)
                        pbar.update(1)

        # Add samples that were not scored (sampling strategy)
        if samples_without_scoring:
            logger.info(
                f"Adding {len(samples_without_scoring)} unscored samples " f"(sampled beyond max_samples_for_scoring)"
            )
            high_quality_samples.extend(samples_without_scoring)

        return high_quality_samples

    def _score_batch_wrapper(
        self, batch: List[Dict[str, str]], task_description: str, batch_idx: int, total_batches: int
    ) -> List[Dict[str, Any]]:
        """Wrapper for batch scoring (used in parallel execution)"""
        return self._score_batch(batch, task_description)

    def _score_batch(self, samples: List[Dict[str, str]], task_description: str) -> List[Dict[str, Any]]:
        """Score a batch of samples using LLM.

        Args:
            samples: List of Alpaca samples (up to batch_size)
            task_description: User's task description

        Returns:
            List of score info: [{"index": 0, "score": 8.5, "reason": "..."}, ...]
        """
        # Prepare samples JSON
        samples_json = json.dumps(
            [
                {
                    "index": i,
                    "instruction": s["instruction"],
                    "input": s["input"],
                    "output": s["output"],
                }
                for i, s in enumerate(samples)
            ],
            ensure_ascii=False,
            indent=2,
        )

        # Load and render prompts using T template system
        sys_prompt = T(".prompts:quality_scoring_batch.system").r()
        user_prompt = T(".prompts:quality_scoring_batch.user").r(
            task_description=task_description or "General SFT training",
            num_samples=str(len(samples)),
            samples_json=samples_json,
        )

        # Call LLM
        response = self.llm_client.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        result = json.loads(response)

        # Validate result
        if "samples" not in result or len(result["samples"]) != len(samples):
            raise ValueError(
                f"Invalid LLM response: expected {len(samples)} scores, " f"got {len(result.get('samples', []))}"
            )

        return result["samples"]
