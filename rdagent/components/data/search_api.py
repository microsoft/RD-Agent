"""
API wrappers for dataset search platforms.
"""

from typing import Any, Dict, List, Optional

from rdagent.log import rdagent_logger as logger


class HuggingFaceSearchAPI:
    """Wrapper for HuggingFace Hub API to search datasets"""

    def __init__(self):
        # Import check only
        try:
            from huggingface_hub import HfApi  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for dataset search. "
                "Install it with: pip install -U 'huggingface_hub[cli]'"
            ) from e

    def search_datasets(
        self,
        domain: Optional[str] = None,
        size_categories: Optional[str] = None,
        language: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace datasets using 3 core dimensions.

        Args:
            domain: Domain/topic keyword for fuzzy text search (e.g., "finance", "medical")
            size_categories: Dataset size range (e.g., "10K<n<100K", "1M<n<10M", null for any)
            language: Language code (e.g., "zh", "en", "multilingual", null for any)
            sort: Sort method ("downloads", "likes", "last_modified")
            limit: Maximum results

        Returns:
            List of dataset info dicts with keys:
            - id: Dataset ID (e.g., "squad")
            - downloads: Download count
            - likes: Like count
            - description: Dataset description
            - tags: List of tags
            - last_modified: Last modified timestamp
        """
        from huggingface_hub import HfApi

        try:
            # Build parameters following HuggingFace official API
            search_params: Dict[str, Any] = {
                "sort": sort,
                "limit": limit,
                "full": True,  # Fetch all dataset metadata
                "gated": False,  # Exclude gated datasets by default
            }

            # Domain → search parameter (fuzzy text matching)
            if domain:
                search_params["search"] = domain

            # Size and Language → filter list (precise tag matching)
            filter_list = []
            if size_categories:
                filter_list.append(f"size_categories:{size_categories}")
            if language:
                filter_list.append(f"language:{language}")

            if filter_list:
                search_params["filter"] = filter_list

            logger.info(
                f"Searching HuggingFace: domain='{domain}', "
                f"size_categories={size_categories}, language={language}"
            )

            # Create fresh API instance for each search
            api = HfApi()
            datasets = api.list_datasets(**search_params)

            # Convert to dict format
            results = []
            for ds in datasets:
                results.append(
                    {
                        "id": ds.id,
                        "downloads": getattr(ds, "downloads", 0),
                        "likes": getattr(ds, "likes", 0),
                        "description": getattr(ds, "description", ""),  # Truncate for LLM input
                        "tags": getattr(ds, "tags", []),  # Limit tags count
                        "last_modified": getattr(ds, "lastModified", None),
                    }
                )
                # Break early if we have enough results
                if len(results) >= limit:
                    break

            logger.info(f"Found {len(results)} datasets")
            return results

        except Exception as e:
            logger.error(f"HuggingFace search failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_id: Dataset ID (e.g., "squad")

        Returns:
            Dataset info dict or None if not found
        """
        from huggingface_hub import HfApi

        try:
            api = HfApi()
            info = api.dataset_info(dataset_id)
            return {
                "id": info.id,
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "description": getattr(info, "description", ""),
                "tags": getattr(info, "tags", []),
                "card_data": getattr(info, "cardData", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
            return None