"""
Autonomous dataset search agent with LLM-powered search and selection.
"""

import json
from typing import Any, Dict, List, Optional

from rdagent.components.data.search_api import HuggingFaceSearchAPI
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.download.hf import download_dataset
from rdagent.utils.agent.tpl import T


class DatasetSearchAgent:
    """
    Autonomous dataset search agent for HuggingFace.

    Workflow:
    1. User provides task description
    2. LLM generates search parameters (domain, size_categories, language)
    3. Search HuggingFace datasets via API with filter mechanism
    4. If no results, retry with progressively relaxed constraints
    5. LLM selects best dataset from candidates (judges task relevance in this step)
    6. Download dataset to local directory
    """

    def __init__(self, api_backend: Optional[APIBackend] = None):
        """
        Args:
            api_backend: LLM API backend. If None, will create default instance.
        """
        self.api = api_backend or APIBackend()
        self.hf_api = HuggingFaceSearchAPI()

    def search_and_download(
        self,
        task_description: str,
        max_candidates: int = 20,
        out_dir: Optional[str] = None,
        max_retries: int = 4,
    ) -> Dict[str, Any]:
        """
        End-to-end workflow: search -> select -> download.

        This method reuses search_only() for all search logic and adds download step.

        Retry strategy:
        - Retry 1: LLM intelligently regenerates parameters
        - Retry 2-4: Rule-based progressive relaxation (size → language)

        Args:
            task_description: User's task requirement description
            max_candidates: Maximum number of candidate datasets to retrieve
            out_dir: Output directory for downloaded dataset
            max_retries: Maximum retry attempts (default: 4)

        Returns:
            {
                "dataset_path": str,           # Local path to downloaded dataset
                "dataset_id": str,             # HuggingFace dataset ID
                "reason": str,                 # Selection reasoning
                "confidence": float,           # LLM's confidence score
                "alternatives": List[str],     # Alternative dataset IDs
                "search_params": Dict,         # Search parameters used
            }

        Raises:
            ValueError: If no suitable dataset found after retries
        """
        logger.info(f"Starting dataset search and download for task: {task_description}")

        # Step 1-3: Reuse search_only() for complete search + select logic
        result = self.search_only(
            task_description=task_description,
            max_candidates=max_candidates,
            max_retries=max_retries,
        )

        # Step 4: Download dataset
        dataset_path = download_dataset(
            repo_id=result["dataset_id"],
            out_dir_root=out_dir,
        )
        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Return result with dataset_path, excluding all_candidates
        return {
            "dataset_path": dataset_path,
            "dataset_id": result["dataset_id"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "alternatives": result["alternatives"],
            "warnings": result.get("warnings", []),
            "search_params": result["search_params"],
        }

    def _generate_search_params(self, task_description: str) -> Dict[str, Any]:
        """
        Use LLM to generate search parameters from task description.

        Returns:
            {
                "domain": str,                    # Domain/topic keyword (single word)
                "size_categories": str or None,   # Dataset size range
                "language": str or None,          # Language code
                "reasoning": str                  # LLM's reasoning
            }
        """
        sys_prompt = T(".prompts:search_params.system").r()
        user_prompt = T(".prompts:search_params.user").r(task_description=task_description)

        try:
            response = self.api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict[str, str | None],  # Allow None for optional fields
            )

            # Parse JSON if response is string
            if isinstance(response, str):
                response = json.loads(response)

            # Validate response
            if not isinstance(response, dict) or "domain" not in response:
                raise ValueError(f"Invalid response format: {response}")

            # Log LLM reasoning
            logger.info(f"LLM reasoning: {response.get('reasoning', 'N/A')}")

            # Remove reasoning from return value (keep it in logs only)
            result = {k: v for k, v in response.items() if k != "reasoning"}
            return result

        except Exception as e:
            logger.error(f"Failed to generate search parameters: {e}")
            # Fallback: use task description as domain
            logger.warning("Using fallback search parameters")
            return {
                "domain": task_description[:100],
                "size_categories": None,
                "language": None,
            }

    def _select_best_dataset(
        self,
        task_description: str,
        candidates: List[Dict[str, Any]],
        user_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to select the best dataset from candidates with 4-dimension evaluation.

        Args:
            task_description: User's task requirement
            candidates: List of candidate datasets
            user_language: User's language preference (from search params)

        Returns:
            {
                "dataset_id": str,
                "reason": str,
                "confidence": float,
                "alternatives": List[str],
                "warnings": List[str] (optional)
            }
        """
        sys_prompt = T(".prompts:dataset_selection.system").r(
            user_language=user_language,
        )
        user_prompt = T(".prompts:dataset_selection.user").r(
            task_description=task_description,
            candidates_json=self._format_candidates(candidates),
            user_language=user_language,
        )

        try:
            response = self.api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict[str, str | float | list],
            )

            # Parse JSON if response is string
            if isinstance(response, str):
                response = json.loads(response)

            # Validate response
            required_keys = ["dataset_id", "reason", "confidence"]
            if not all(k in response for k in required_keys):
                raise ValueError(f"Missing required keys in response: {response}")

            return response

        except Exception as e:
            logger.error(f"Failed to select dataset: {e}")
            # Fallback: select first candidate with highest downloads
            logger.warning("Using fallback selection (highest downloads)")
            best = max(candidates, key=lambda x: x.get("downloads", 0))
            return {
                "dataset_id": best["id"],
                "reason": f"Fallback selection: highest downloads ({best.get('downloads', 0)})",
                "confidence": 0.5,
                "alternatives": [],
            }

    def _llm_retry_params(
        self,
        task_description: str,
        failed_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use LLM to intelligently generate new search parameters after failure.

        This method is called on the FIRST retry attempt. The LLM is given
        strategic guidance to try domain synonyms and relax constraints.

        Args:
            task_description: Original user request
            failed_params: Parameters that returned 0 results

        Returns:
            New search parameters generated by LLM with strategic adjustments

        Raises:
            Falls back to rule-based relaxation if LLM fails
        """
        sys_prompt = T(".prompts:retry_search_params.system").r()
        user_prompt = T(".prompts:retry_search_params.user").r(
            task_description=task_description,
            failed_params_json=json.dumps(failed_params, indent=2, ensure_ascii=False),
        )

        try:
            response = self.api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=dict[str, str | None],
            )

            # Parse JSON if response is string
            if isinstance(response, str):
                response = json.loads(response)

            # Validate response
            if not isinstance(response, dict) or "domain" not in response:
                raise ValueError(f"Invalid LLM retry response: {response}")

            # Log LLM reasoning
            logger.info(f"LLM retry reasoning: {response.get('reasoning', 'N/A')}")

            # Remove reasoning from return value
            result = {k: v for k, v in response.items() if k != "reasoning"}
            return result

        except Exception as e:
            logger.error(f"LLM retry failed: {e}, falling back to rule-based relaxation")
            # Fallback: if LLM retry fails, use simple rule-based relaxation
            return self._relax_search_params(failed_params)

    def _relax_search_params(self, original_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based parameter relaxation (used after LLM retry fails).

        This method is called on retry attempts AFTER the first LLM retry.
        It progressively removes constraints in order of importance.

        Strategy (in priority order):
        1. Remove size_categories filter (least critical - dataset size)
        2. Remove language filter (secondary - cross-lingual datasets may work)

        Args:
            original_params: Original search parameters

        Returns:
            Relaxed search parameters with one constraint removed

        Raises:
            ValueError: If no further relaxation is possible (all filters removed)
        """
        relaxed = original_params.copy()

        # Step 1: Remove size_categories filter (least critical)
        if relaxed.get("size_categories"):
            logger.info("Rule-based retry: Removing size_categories filter")
            relaxed["size_categories"] = None
            return relaxed

        # Step 2: Remove language filter (secondary importance)
        if relaxed.get("language"):
            logger.info("Rule-based retry: Removing language filter")
            relaxed["language"] = None
            return relaxed

        # Step 3: No more relaxation possible - all filters removed
        logger.error("No further relaxation possible - all filters already removed")
        raise ValueError(
            f"Search failed after exhausting all retry strategies. "
            f"Last search params: domain='{relaxed.get('domain')}', all filters removed"
        )

    def _apply_license_blacklist(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out datasets with non-commercial or restrictive licenses.

        Blacklist includes:
        - NC (Non-Commercial): cc-by-nc, cc-by-nc-sa, cc-by-nc-nd
        - ND (No Derivatives): cc-by-nd
        - Copyleft: gpl, agpl (requires derivative works to be open-sourced)

        Args:
            candidates: List of candidate datasets

        Returns:
            Filtered list excluding blacklisted licenses
        """
        BLACKLIST_LICENSES = [
            "cc-by-nc",
            "cc-by-nc-sa",
            "cc-by-nc-nd",
            "cc-by-nd",
            "gpl",
            "agpl",
        ]

        filtered = []
        for c in candidates:
            tags = c.get("tags", [])
            license_tags = [t for t in tags if t.startswith("license:")]

            is_blacklisted = False
            for lic_tag in license_tags:
                # Extract license value (e.g., "license:cc-by-nc" -> "cc-by-nc")
                lic_value = lic_tag.split(":", 1)[1].lower()

                # Check if any blacklist term is in the license
                if any(bl in lic_value for bl in BLACKLIST_LICENSES):
                    logger.info(f"⚠️  Filtered out {c['id']} due to restrictive license: {lic_value}")
                    is_blacklisted = True
                    break

            if not is_blacklisted:
                filtered.append(c)

        logger.info(f"License filter: {len(candidates)} candidates → {len(filtered)} after blacklist")
        return filtered

    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Format candidate datasets for LLM consumption with complete tags and structured info.

        Provides both raw tags and extracted structured information:
        - tags: Complete list of all HuggingFace tags (including task_categories, size_categories, etc.)
        - structured_info: Pre-extracted common fields for quick reference
          - task_ids: Task categories (e.g., question-answering, summarization)
          - languages: Language tags (e.g., en, zh)
          - licenses: License types (e.g., mit, apache-2.0)
          - modalities: Data modalities (e.g., text, image)

        Returns:
            JSON string with candidate information including both tags and structured_info
        """
        formatted = []
        for c in candidates:
            tags = c.get("tags", [])

            # Extract structured information from tags for quick reference
            task_ids = [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")]
            languages = [t.split(":", 1)[1] for t in tags if t.startswith("language:")]
            licenses = [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
            modalities = [t.split(":", 1)[1] for t in tags if t.startswith("modality:")]

            formatted.append(
                {
                    "id": c["id"],
                    "downloads": c.get("downloads", 0),
                    "likes": c.get("likes", 0),
                    "description": c.get("description", ""),
                    "tags": tags,  # Complete raw tags for full context
                    "structured_info": {
                        "task_ids": task_ids,
                        "languages": languages,
                        "licenses": licenses,
                        "modalities": modalities,
                    },
                }
            )
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    def search_only(
        self,
        task_description: str,
        max_candidates: int = 20,
        max_retries: int = 4,
    ) -> Dict[str, Any]:
        """
        Search and select dataset without downloading.

        Retry strategy:
        - Retry 1: LLM intelligently regenerates parameters
        - Retry 2-4: Rule-based progressive relaxation (size → language)

        Args:
            task_description: User's task requirement description
            max_candidates: Maximum number of candidate datasets to retrieve
            max_retries: Maximum retry attempts (default: 4)

        Returns:
            {
                "dataset_id": str,
                "reason": str,
                "confidence": float,
                "alternatives": List[str],
                "search_params": Dict,
                "all_candidates": List[Dict]
            }

        Raises:
            ValueError: If no datasets found after all retries
        """
        logger.info(f"Search-only mode for task: {task_description}")

        # Step 1: Generate initial search parameters
        search_params = self._generate_search_params(task_description)
        logger.info(f"Initial search parameters: {search_params}")

        # Step 2: Search with hybrid retry mechanism
        candidates = None
        retry_count = 0
        used_llm_retry = False  # Track if LLM retry has been used

        while candidates is None or len(candidates) == 0:
            if retry_count > max_retries:
                raise ValueError(
                    f"Failed to find datasets after {max_retries} retries. " f"Last search params: {search_params}"
                )

            # Attempt search
            candidates = self.hf_api.search_datasets(
                domain=search_params.get("domain"),
                size_categories=search_params.get("size_categories"),
                language=search_params.get("language"),
                limit=max_candidates,
            )

            if not candidates:
                logger.warning(
                    f"No results found (attempt {retry_count + 1}/{max_retries + 1}). "
                    "Retrying with adjusted parameters..."
                )

                # === Key Logic: LLM retry on first failure, rule-based after ===
                if retry_count == 0 and not used_llm_retry:
                    # Retry 1: LLM intelligent retry
                    logger.info("🤖 Using LLM to intelligently adjust search parameters")
                    search_params = self._llm_retry_params(task_description, search_params)
                    used_llm_retry = True
                else:
                    # Retry 2-4: Rule-based relaxation
                    logger.info("📋 Using rule-based parameter relaxation")
                    search_params = self._relax_search_params(search_params)

                retry_count += 1
            else:
                logger.info(f"✅ Found {len(candidates)} candidate datasets")

        # Step 2.5: Apply license blacklist filter
        candidates = self._apply_license_blacklist(candidates)

        if not candidates:
            raise ValueError(
                "All candidate datasets were filtered out due to restrictive licenses (NC/ND/GPL). "
                "Try relaxing search parameters or manually selecting a dataset."
            )

        # Step 3: Select best dataset with 4-dimension evaluation
        selection = self._select_best_dataset(
            task_description=task_description,
            candidates=candidates,
            user_language=search_params.get("language"),
        )

        # Log warnings if any
        warnings = selection.get("warnings", [])
        if warnings:
            logger.warning(f"Dataset selection warnings: {warnings}")

        return {
            "dataset_id": selection["dataset_id"],
            "reason": selection["reason"],
            "confidence": selection["confidence"],
            "alternatives": selection.get("alternatives", []),
            "warnings": warnings,
            "search_params": search_params,
            "all_candidates": candidates,
        }
