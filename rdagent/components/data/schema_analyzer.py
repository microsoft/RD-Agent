"""Schema analyzer for SFT dataset conversion.

Analyzes dataset structure to determine:
1. Data type (single-turn Q&A vs multi-turn dialogue)
2. Column name mapping for Alpaca format conversion

Fully autonomous LLM-driven analysis with intelligent retry mechanism.
"""

import json
import logging
from typing import Any

import pandas as pd

from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow.misc import wait_retry

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """Analyze dataset schema for SFT conversion using LLM-driven approach."""

    def __init__(self, llm_client):
        """Initialize schema analyzer.

        Args:
            llm_client: LLM client for schema analysis
        """
        self.llm_client = llm_client

    @wait_retry(retry_n=3)
    def analyze(self, sample_data: pd.DataFrame, task_description: str) -> dict[str, Any]:
        """Analyze dataset schema using LLM (fully autonomous).

        Args:
            sample_data: Sample DataFrame (first 10-100 rows)
            task_description: User's task description

        Returns:
            Schema analysis result:
            {
                "data_type": "single_turn" | "multi_turn",
                "instruction_col": column name,
                "input_col": column name or None,
                "output_col": column name,
                "reasoning": explanation
            }
        """
        logger.info("Analyzing dataset schema with LLM...")

        # Prepare sample data (first 10 rows)
        sample_size = min(10, len(sample_data))
        sample_dict = sample_data.head(sample_size).to_dict(orient="records")

        # Load and render prompts using T template system
        sys_prompt = T(".prompts:schema_analysis_for_sft.system").r()
        user_prompt = T(".prompts:schema_analysis_for_sft.user").r(
            task_description=task_description,
            column_names=str(list(sample_data.columns)),
            sample_data_json=json.dumps(sample_dict, ensure_ascii=False, indent=2),
        )

        # Call LLM
        response = self.llm_client.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        result = json.loads(response)

        # Validate result
        self._validate_schema_result(result, sample_data.columns)

        logger.info(f"Schema analysis successful: {result}")
        return result

    def _validate_schema_result(self, result: dict[str, Any], columns: list) -> None:
        """Validate LLM schema analysis result.

        Args:
            result: Schema analysis result from LLM
            columns: Actual column names in dataset

        Raises:
            ValueError: If result is invalid
        """
        # Check required fields
        required_fields = ["data_type", "instruction_col", "output_col"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Check data_type value
        if result["data_type"] not in ["single_turn", "multi_turn"]:
            raise ValueError(f"Invalid data_type: {result['data_type']}")

        # Check if instruction_col and output_col exist in columns
        if result["instruction_col"] not in columns:
            raise ValueError(f"instruction_col '{result['instruction_col']}' not in columns: {columns}")

        if result["output_col"] not in columns:
            raise ValueError(f"output_col '{result['output_col']}' not in columns: {columns}")

        # Check input_col if specified
        if result.get("input_col") and result["input_col"] not in columns:
            raise ValueError(f"input_col '{result['input_col']}' not in columns: {columns}")
