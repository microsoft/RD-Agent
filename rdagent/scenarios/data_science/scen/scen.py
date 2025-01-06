import json
import os
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


def read_csv_head(file_path, indent, lines=5):
    try:
        df = pd.read_csv(file_path, nrows=lines)
        df_string_lines = df.to_string(index=False).split("\n")
        for i in range(len(df_string_lines)):
            df_string_lines[i] = " " * (indent) + df_string_lines[i]
        return "\n".join(df_string_lines)
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


def describe_data_folder(folder_path, indent=0, max_files=3, partial_expand_subfolders=3):
    """
    folder_path              : Current directory path
    indent                   : Current indentation
    max_files                : Maximum number of files of the same type to display
    partial_expand_subfolders: When all subfolders have the same internal file types, only expand this many subfolders, the rest are omitted
    """
    result = []
    files_count = {}
    files_details = {}

    for root, dirs, files in os.walk(folder_path):
        dirs.sort()

        if not dirs:
            for file in files:
                file_path = os.path.join(root, file)
                file_type = os.path.splitext(file)[1][1:]
                file_size = os.path.getsize(file_path)

                if file_type not in files_count:
                    files_count[file_type] = 0
                    files_details[file_type] = []
                files_count[file_type] += 1
                if len(files_details[file_type]) < max_files:
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
                            partial_expand_subfolders=partial_expand_subfolders
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
                        partial_expand_subfolders=partial_expand_subfolders
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

            if len(files_details[file_type]) < max_files:
                files_details[file_type].append((file, file_size, file_path))

        break

    # Print the folder and its contents
    for file_type, count in files_count.items():
        if count > max_files:
            result.append(" " * indent + f"{count} {file_type}s:")
            for file, size, path in files_details[file_type]:
                result.append(" " * (indent + 2) + f"- {file} ({size} bytes)")
            result.append(" " * (indent + 2) + "... (file limit reached)")
        else:
            for file, size, path in files_details[file_type]:
                if file_type == "zip":
                    continue
                result.append(" " * indent + f"- {file} ({size} bytes)")
                if file_type == "csv":
                    result.append(" " * (indent + 2) + f"- Head of {file}:")
                    result.append(read_csv_head(path, indent + 2))
                if file_type == "md":
                    result.append(" " * (indent + 2) + f"- Content of {file}:")
                    with open(path, "r", encoding="utf-8") as f:
                        result.append(f.read())

    return "\n".join(result) + "\n"


class DataScienceScen(Scenario):
    """Data Science Scenario"""

    def __init__(self, competition: str) -> None:
        self.competition = competition
        self.raw_description = self._get_description()
        self.metric_direction = self._get_direction()
        self._analysis_competition_description()

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
        return self.raw_description.get("metric_direction", "minimize")

    def _analysis_competition_description(self):
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.raw_description,
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
        self.data_description = response_json_analysis.get("Data Description", "No data description provided")
        self.target_description = response_json_analysis.get("Target Description", "No target description provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)

    def get_competition_full_desc(self) -> str:
        return f"""Task Type: {self.task_type}
    Data Type: {self.data_type}
    Brief Description: {self.brief_description}
    Data Description: {self.data_description}
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
            data_description=self.data_description,
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
            metric_direction=self.metric_direction,
        )

    def get_data_folder_description(self) -> str:
        return describe_data_folder(Path(DS_RD_SETTING.local_data_path) / self.competition)
