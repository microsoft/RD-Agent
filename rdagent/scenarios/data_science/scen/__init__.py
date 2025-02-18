import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image, TiffTags

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.kaggle_crawler import (
    crawl_descriptions,
    leaderboard_scores,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf


def read_csv_head(file_path, indent=0, lines=5, max_col_width=100):
    """
    Reads the first few rows of a CSV file and formats them with indentation and optional truncation.

    Parameters:
        file_path (str): Path to the CSV file.
        indent (int): Number of spaces to prepend to each line for indentation.
        lines (int): Number of rows to read from the CSV file.
        max_col_width (int): Maximum width of each column's content.

    Returns:
        str: A formatted string of the first few rows of the CSV file.
    """
    try:
        # Read the CSV file with specified rows
        df = pd.read_csv(file_path, nrows=lines)

        if df.empty:
            return " " * indent + "(No data in the file)"

        # Truncate column contents to a maximum width
        truncated_df = df.copy()
        for col in truncated_df.columns:
            truncated_df[col] = (
                truncated_df[col]
                .astype(str)
                .apply(lambda x: (x[:max_col_width] + "...") if len(x) > max_col_width else x)
            )

        # Convert DataFrame to a string representation
        df_string_lines = truncated_df.to_string(index=False).split("\n")

        # Add indentation to each line
        indented_lines = [" " * indent + line for line in df_string_lines]

        return "\n".join(indented_lines)
    except FileNotFoundError:
        return f"Error: File not found at path '{file_path}'."
    except pd.errors.EmptyDataError:
        return f"Error: The file at '{file_path}' is empty."
    except Exception as e:
        return f"Error reading CSV: {e}"


def get_dir_snapshot(folder_path):
    """
    [note]
        - Returns a set of file extensions within the subfolder (excluding subfolder names)
        - Compares only the types of files contained, not specific file names or quantities
    """
    exts = set()
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file():
                    file_ext = os.path.splitext(entry.name)[1]
                    exts.add(file_ext)
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")

    return frozenset(exts)


def describe_data_folder(folder_path, indent=0, max_files=2, partial_expand_subfolders=2, is_top_level=True):
    """
    folder_path              : Current directory path
    indent                   : Current indentation
    max_files                : Maximum number of files of the same type to display
    partial_expand_subfolders: When all subfolders have the same internal file types, only expand this many subfolders, the rest are omitted
    is_top_level             : Indicates if the current folder is the top-level folder
    """
    result = []
    files_count = {}
    files_details = {}

    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        if not dirs:
            for file in files:
                file_path = os.path.join(root, file)
                file_type = os.path.splitext(file)[1][1:]
                file_size = os.path.getsize(file_path)

                if file_type not in files_count:
                    files_count[file_type] = 0
                    files_details[file_type] = []
                files_count[file_type] += 1

                # At top level, collect all CSV and Markdown files without restrictions
                # In deeper levels, follow the max_files restriction
                if is_top_level and file_type in ["csv", "md"]:
                    files_details[file_type].append((file, file_size, file_path))
                elif len(files_details[file_type]) < max_files:
                    files_details[file_type].append((file, file_size, file_path))
            break

        # Collect "type snapshots" of subfolders
        snapshots = []
        for d in dirs:
            subfolder_path = os.path.join(root, d)
            snapshot = get_dir_snapshot(subfolder_path)
            snapshots.append(snapshot)

        # Determine if all subfolders have the same file type distribution
        first_snapshot = snapshots[0]
        all_same_structure = all(s == first_snapshot for s in snapshots)

        if all_same_structure:
            for i, d in enumerate(dirs):
                if i < partial_expand_subfolders:
                    result.append(" " * indent + f"- Folder: {d}")
                    subfolder_path = os.path.join(root, d)
                    result.append(
                        describe_data_folder(
                            folder_path=subfolder_path,
                            indent=indent + 2,
                            max_files=max_files,
                            partial_expand_subfolders=partial_expand_subfolders,
                            is_top_level=False,
                        )
                    )
                else:
                    remaining = len(dirs) - i
                    result.append(" " * indent + f"... ({remaining} more subfolders)")
                    break
        else:
            for d in dirs:
                result.append(" " * indent + f"- Folder: {d}")
                subfolder_path = os.path.join(root, d)
                result.append(
                    describe_data_folder(
                        folder_path=subfolder_path,
                        indent=indent + 2,
                        max_files=max_files,
                        partial_expand_subfolders=partial_expand_subfolders,
                        is_top_level=False,
                    )
                )

        for file in files:
            file_path = os.path.join(root, file)
            file_type = os.path.splitext(file)[1][1:]
            file_size = os.path.getsize(file_path)

            if file_type not in files_count:
                files_count[file_type] = 0
                files_details[file_type] = []
            files_count[file_type] += 1

            # At top level, collect all CSV and Markdown files without restrictions
            # In deeper levels, follow the max_files restriction
            if is_top_level and file_type in ["csv", "md"]:
                files_details[file_type].append((file, file_size, file_path))
            elif not is_top_level and len(files_details[file_type]) <= max_files:
                files_details[file_type].append((file, file_size, file_path))

        break

    # Print the folder and its contents
    for file_type, count in files_count.items():
        if count > max_files and file_type not in ["csv", "md", "txt"]:
            result.append(" " * indent + f"{count} {file_type}s:")
            for file, size, path in files_details[file_type]:
                result.append(" " * (indent + 2) + f"- {file} ({size} bytes)")
            result.append(" " * (indent + 2) + "... (file limit reached)")
        else:
            for file, size, path in files_details[file_type]:
                if file_type == "csv":
                    df = pd.read_csv(path)
                    result.append(
                        " " * indent + f"- {file} ({size} bytes, with {df.shape[0]} rows and {df.shape[1]} columns)"
                    )
                    result.append(" " * (indent + 2) + f"- Head of {file}:")
                    csv_head = read_csv_head(path, indent + 4)
                    result.append(csv_head)
                    continue
                result.append(" " * indent + f"- {file} ({size} bytes)")
                if file_type == "md":
                    result.append(" " * (indent + 2) + f"- Content of {file}:")
                    if file == "description.md":
                        result.append(" " * (indent + 4) + f"Please refer to the background of the scenario context.")
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        result.append(" " * (indent + 4) + f.read())
                if file_type == "tif":
                    result.append(" " * (indent + 2) + f"- Metadata of {file}:")
                    with Image.open(path) as img:
                        for tag, value in img.tag_v2.items():
                            tag_name = TiffTags.TAGS_V2.get(tag, f"Unknown Tag {tag}")
                            result.append(" " * (indent + 4) + f"{tag_name}: {value}")
                if file_type in ["json", "txt"]:
                    result.append(" " * (indent + 2) + f"- Content of {file}:")
                    with open(path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i < 2:
                                result.append(
                                    " " * (indent + 4) + line.strip()[:100] + ("..." if len(line.strip()) > 100 else "")
                                )
                            else:
                                break

    return "\n".join(result) + "\n"


class DataScienceScen(Scenario):
    """Data Science Scenario"""

    def __init__(self, competition: str) -> None:
        self.competition = competition
        self.raw_description = self._get_description()
        self.processed_data_folder_description = self._get_data_folder_description()
        self._analysis_competition_description()
        self.metric_direction = self._get_direction()

    def _get_description(self):
        if (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}.json")).exists():
            logger.info(f"Found {self.competition}.json, loading from local file.")
            with fp.open("r") as f:
                return json.load(f)
        else:
            logger.error(
                f"Cannot find {self.competition}.json in {DS_RD_SETTING.local_data_path}, please check the file."
            )

    def _get_direction(self):
        return self.metric_direction_guess if hasattr(self, "metric_direction_guess") else True

    def _analysis_competition_description(self):
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.raw_description,
            competition_processed_data_folder_description=self.processed_data_folder_description,
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json_analysis = json.loads(response_analysis)
        self.task_type = response_json_analysis.get("Task Type", "No type provided")
        self.data_type = response_json_analysis.get("Data Type", "No data type provided")
        self.brief_description = response_json_analysis.get("Brief Description", "No brief description provided")
        self.dataset_description = response_json_analysis.get("Dataset Description", "No dataset description provided")
        self.target_description = response_json_analysis.get("Evaluation Description", "No target description provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)
        self.metric_direction_guess = response_json_analysis.get("Metric Direction", True)

    def get_competition_full_desc(self) -> str:
        return f"""Task Type: {self.task_type}
    Data Type: {self.data_type}
    Brief Description: {self.brief_description}
    Dataset Description: {self.dataset_description}
    Target Description: {self.target_description}
    Submission Specifications: {self.submission_specifications}
    Model Output Channel: {self.model_output_channel}
    """

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            task_type=self.task_type,
            data_type=self.data_type,
            brief_description=self.brief_description,
            dataset_description=self.dataset_description,
            target_description=self.target_description,
        )
        return background_prompt

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Data Science",
            competition=self.competition,
        )

    def get_scenario_all_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            background=self.background,
            submission_specifications=self.submission_specifications,
            evaluation=self.target_description,
            metric_direction=self.metric_direction,
        )

    def get_runtime_environment(self) -> str:
        # TODO:  add it into base class.  Environment should(i.e. `DSDockerConf`) should be part of the scenario class.
        ds_docker_conf = DSDockerConf()
        de = DockerEnv(conf=ds_docker_conf)
        implementation = FBWorkspace()
        fname = "temp.py"
        implementation.inject_files(
            **{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()}
        )
        stdout = implementation.execute(env=de, entry=f"python {fname}")
        return stdout

    def _get_data_folder_description(self) -> str:
        return describe_data_folder(Path(DS_RD_SETTING.local_data_path) / self.competition)


class KaggleScen(DataScienceScen):
    """Kaggle Scenario
    It is based on kaggle now.
        - But it is not use the same interface with previous kaggle version.
        - Ideally, we should reuse previous kaggle scenario.
          But we found that too much scenario unrelated code in kaggle scenario and hard to reuse.
          So we start from a simple one....
    """

    def _get_description(self):
        return crawl_descriptions(self.competition, DS_RD_SETTING.local_data_path)

    def _get_direction(self):
        if DS_RD_SETTING.if_using_mle_data:
            return super()._get_direction()
        leaderboard = leaderboard_scores(self.competition)
        return "maximize" if float(leaderboard[0]) > float(leaderboard[-1]) else "minimize"

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Kaggle",
            competition=f"[{self.competition}](https://www.kaggle.com/competitions/{self.competition})",
        )


if __name__ == "__main__":
    print(describe_data_folder(Path("/data/userdata/share/mle_kaggle") / "stanford-covid-vaccine"))
