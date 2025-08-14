import os
from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.utils.agent.tpl import T


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self, dataset: str) -> None:
        """Initialize LLM finetune scenario without calling base __init__.

        Only keep logic needed for LLM finetuning datasets.
        """
        local_path = FT_RD_SETTING.local_data_path
        dataset_path = Path(local_path) / dataset

        # 1) Ensure dataset exists (download if missing)
        if not dataset_path.exists():
            logger.info(f"{dataset_path} does not exist. Try to download from hub.")
            self._download_data(competition=dataset)

        # 1.1) Ensure prev_model and model are visible under dataset_path for mounting to workspace_input
        try:
            ft_root = Path(local_path).parent  # FT_FILE_PATH
            model_name = FT_RD_SETTING.base_model_name
            # Expected: prev_model/<baseModel_dataset>/, model/<baseModel>/
            prev_src = (
                ft_root / "prev_model" / f"{model_name}_{dataset}" if model_name else ft_root / "prev_model" / dataset
            )
            model_src = ft_root / "model" / model_name if model_name else None
            prev_dst = dataset_path / "prev_model"
            model_dst = dataset_path / "model"

            def _ensure_link(src: Path, dst: Path) -> None:
                if src.exists():
                    if dst.is_symlink() or dst.exists():
                        return
                    try:
                        dst.symlink_to(src, target_is_directory=True)
                    except Exception:
                        # Fallback to creating the directory if symlink not allowed
                        if src.is_dir():
                            # Do not copy content here to keep cost low; rely on direct mount via dataset folder
                            pass

            # Prefer prev_model; if missing and model exists, expose model as prev_model for compatibility
            if prev_src.exists():
                _ensure_link(prev_src, prev_dst)
            elif model_src is not None and model_src.exists():
                _ensure_link(model_src, prev_dst)
            # Also expose model separately if exists
            if model_src is not None and model_src.exists():
                _ensure_link(model_src, model_dst)
        except Exception:
            pass

        # 2) Basic attributes (align with downstream expectations)
        self.competition = dataset  # keep field name for downstream compatibility
        self.metric_name = None

        # 3) Debug path: use the dataset folder as mount root so `prev_model/` symlink is inside it
        self.debug_path = str(dataset_path)

        # 4) Description and folder summary
        self.raw_description = self._get_description()
        self.processed_data_folder_description = self._get_data_folder_description()

        # 5) Analyze dataset description (override of competition analysis)
        self._analysis_dataset_description()

        # 6) timeout tracking
        self.timeout_increase_count = 0

    def real_debug_timeout(self):
        return FT_RD_SETTING.debug_timeout

    def recommend_debug_timeout(self):
        return FT_RD_SETTING.debug_recommend_timeout

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def recommend_full_timeout(self):
        return FT_RD_SETTING.full_recommend_timeout

    @property
    def dataset(self) -> str:
        # Align naming for LLM scenario; the base class uses `competition`.
        return self.competition

    def _get_data_folder_description(self) -> str:
        folder_desc = describe_data_folder_v2(
            Path(FT_RD_SETTING.local_data_path) / self.competition,
            show_nan_columns=FT_RD_SETTING.show_nan_columns,
        )
        return folder_desc

    def _download_data(self, competition: str):
        """
        Download dateset from Hugging Face Hub

        Parameters
        ----------
        - competition (str): Dateset ID, like "shibing624/alpaca-zh".
        """
        save_path = f"{FT_RD_SETTING.local_data_path}/{competition}"
        if Path(save_path).exists():
            logger.info(f"{save_path} already exists.")
        else:
            logger.info(f"Downloading {competition} to {save_path}")
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=competition,
                    repo_type="dataset",
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                )
            except ImportError:
                raise ImportError(
                    "Please install huggingface_hub first. "
                    'You can install it with `pip install -U "huggingface_hub[cli]"`.'
                )
            except Exception as e:
                logger.error(f"Error when downloading {competition}: {e}")
                raise e

    def _get_description(self):
        if (fp := Path(f"{FT_RD_SETTING.local_data_path}/{self.competition}/README.md")).exists():
            logger.info(f"{self.competition}/Found README.md, loading from local file.")
            return fp.read_text()

    # ===== use dataset analysis instead of competition analysis =====
    def _analysis_dataset_description(self):
        sys_prompt = T(".prompts:dataset_description_template.system").r()
        user_prompt = T(".prompts:dataset_description_template.user").r(
            raw_description=self.raw_description,
            data_folder_description=self._get_data_folder_description(),
        )
        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=dict,
        )
        # keep the same fields as DataScienceScen for downstream reuse
        try:
            import json

            parsed = json.loads(response_analysis)
        except Exception:
            parsed = {}

        self.task_type = parsed.get("Task Type", "LLM Fine-tuning")
        self.data_type = parsed.get("Data Type", "Text (Natural Language)")
        self.brief_description = parsed.get("Brief Description", "")
        self.dataset_description = parsed.get("Dataset Description", "")
        self.model_output_channel = parsed.get("Channels per Sample", 1)
        self.metric_description = parsed.get("Evaluation Metric Description", "")
        self.metric_name = parsed.get("Metric Name", "custom_metric")
        self.metric_direction_guess = parsed.get("Metric Direction", True)

    @property
    def rich_style_description(self) -> str:
        raise NotImplementedError

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            raw_description=self.raw_description,
        )
        return background_prompt

    def get_competition_full_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """
        eda_output depends on dynamic .md files from current workspace, not fixed.
        """
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )
