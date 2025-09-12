import json
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import fire
import nbformat
import requests
from kaggle.api.kaggle_api_extended import (
    ApiCreateCodeSubmissionResponse,
    ApiSaveKernelResponse,
    KaggleApi,
    SubmissionSortBy,
    SubmissionStatus,
)
from kagglesdk.competitions.types.competition_api_service import (
    ApiCompetition,
    ApiSubmission,
)
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
from pydantic_settings import SettingsConfigDict
from rdagent.core.conf import ExtendedBaseSettings
from rdagent.log import rdagent_logger as logger

# error 403 means we have not join the competition when retrieving information, like submissions
ERROR_CODE_NOT_JOIN_COMPETITION = 403
# error code when we try to get submission when there is not any
ERROR_CODE_NO_SUBMISSION = 400


class KaggleSubmissionSetting(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="SUBMIT_", protected_namespaces=())

    # if force to submit code even it is not a code competition
    force_submit_code: bool = False

    # submission file name to submit
    submission_file: str = "submission.csv"

    # if enable notebook gpu
    enable_gpu: bool = False

    # if enable notebook internet access
    enable_internet: bool = False

    # if enable notebook tpu
    enable_tpu: bool = False

    # if make kernel private
    is_private: bool = True

    # prefix of the notebook name
    kernel_prefix: str = "rd-submission"

    # timeout when tracking submission status
    status_timeout: int = 120


KG_SUBMISSION_SETTING = KaggleSubmissionSetting()


def get_completed_submissions(api: KaggleApi, competition: str) -> list[ApiSubmission]:
    """Get the completed submissions of today.

    Args:
        api (KaggleApi): kaggle api object
        competition (str): competition name to get submissions

    Return:
        list[ApiSubmission]: completed submissions of today
    """
    # only completed submission will consume today's limitation
    submissions: list[ApiSubmission] = []

    today = datetime.now(timezone.utc).date()

    try:
        submissions_resp = api.competition_submissions(
            competition=competition,
            # the api use date to sort the result by default, so latest submission is in last page
            page_token=-1,
        )

        if submissions_resp is not None:
            submissions = [
                submission
                for submission in submissions_resp
                if submission is not None
                and submission.date.date() == today
                and submission.status == SubmissionStatus.COMPLETE
            ]
    except requests.exceptions.HTTPError as e:
        # 403 if we have not joined the competition
        if e.response.status_code == ERROR_CODE_NOT_JOIN_COMPETITION:
            logger.error(f"You have not joined the competition '{competition}'.")
        elif e.response.status_code == ERROR_CODE_NO_SUBMISSION:
            # if a competition has no any submission, this call will cause 400.
            # we consider it is a correct state, means 0 completed submissions
            logger.info("no submission now.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Fail to get submissions with error: {e}")

    return submissions


def get_competition_detail(api: KaggleApi, competition: str) -> ApiCompetition | None:
    """Get the competition detail information.

    Args:
        api (KaggleApi): kaggle api object
        competition (str): competition name to get detail

    Return:
        ApiCompetition: competition detail information
    """
    # kaggle sdk do not have function to get detail for specified competition, here we use the list function
    # list function only return in-progress competitions,
    # and we will use search parameter to filter competition name,
    # and default page_size is 20, so more time we can get the target competition at first page
    max_pages = 10
    try:
        page = 1
        while page <= max_pages and (search_result := api.competitions_list(search=competition, page=page)) is not None:
            for comp in search_result:
                if comp is not None and comp.ref.rsplit("/", 1)[-1] == competition:
                    return comp

            page += 1

            time.sleep(1)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Fail to get competition list, with exception: {e}")

    return None


def get_submission_remaining(api: KaggleApi, competition: str) -> int:
    """Get the submission remaining  number.

    NOTE: this function will try to avoid any exception raised, so if you have not joined the competition,
        or any other exceptions, will return 0

    Args:
        api (KaggleApi): kaggle api object
        competition (str): competition name to get submission remaining number

    Return:
        int: submission remaining number
    """
    # kaggle use utc time to count
    completed_submissions = get_completed_submissions(api, competition)

    competition_detail = get_competition_detail(api, competition)

    if competition_detail is None:
        logger.error("Fail to get the competition, make sure it is in progress now.")

        return 0

    return competition_detail.max_daily_submissions - len(completed_submissions)


def wait_for_submission_complete(api: KaggleApi, competition: str) -> None:
    """Wait for the latest submission complete (failed or completed).

    Args:
        api (KaggleApi): kaggle api object
        competition (str): competition name to wait for submission complete
    """
    timeout = KG_SUBMISSION_SETTING.status_timeout

    # the submission request is done without error, here we keep check the latest submission state, until completed or timeout
    start = datetime.now()  # noqa: DTZ005
    while (datetime.now() - start).seconds <= timeout:  # noqa: DTZ005
        # the api can sort the submissions by date (new -> old), so we use the first item at first page
        resp = api.competition_submissions(
            competition=competition,
            sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
            page_token=0,
        )

        logger.info(f"Current submissions: {resp}")

        if resp is None or resp[0] is None or resp[0].status == SubmissionStatus.PENDING:
            # wait for 5 seconds to get latest result
            time.sleep(5)

            continue

        submission = resp[0]

        if submission.status == SubmissionStatus.ERROR:
            logger.error(f"The submission validation failed, with message: {submission.error_description}")
        else:
            logger.info(
                f"Current submission state [@{submission.date}] is: {submission.status}, public score: {submission.public_score}",
            )

        break


def submit_local_file(api: KaggleApi, competition: str, file: str | Path, *, msg: str = "Message") -> None:
    """Submit local file to competition.

    Args:
        api (KaggleApi): kaggle api object
        competition (str | Path): competition name to submit
        file (str): local file path to submit
        msg (str): message of current submission
    """

    # submit first
    try:
        resp = api.competition_submit(file_name=str(file), message=msg, competition=competition)

        # if upload failed, will return a string message, or it will raise exception if there is http response code >= 400
        if type(resp) is str:
            logger.error(f"Fail to get submissions with error: {resp}")
        else:
            wait_for_submission_complete(api=api, competition=competition)
    except Exception as e:  # noqa: BLE001
        if type(e) is requests.exceptions.HTTPError and e.response.status_code == ERROR_CODE_NOT_JOIN_COMPETITION:
            logger.error(f"You have not joined the competition '{competition}'.")
        else:
            logger.error(f"Fail to get submissions with error: {e}")


def submit_kernel_by_version(
    api: KaggleApi,
    competition: str,
    kernel_id: str,
    kernel_version: int,
    *,
    msg: str = "Message",
) -> None:
    """Submit output of online kernel to competition without downloading the outputs.

    Args:
        api (KaggleApi): kaggle api object
        competition (str): competition name to submit
        kernel_id (str): kernel id to submit
        kernel_version (str): kernel version to submit,
            each time we call kaggle sdk to upload a notebook, it will return a new version
        msg (str): message of current submission
    """
    submission_file = KG_SUBMISSION_SETTING.submission_file

    try:
        resp: ApiCreateCodeSubmissionResponse = api.competition_submit_code(
            competition=competition,
            file_name=submission_file,
            message=msg,
            kernel=kernel_id,
            kernel_version=kernel_version,
        )

        logger.info(f"Submission response: {resp}")

        wait_for_submission_complete(api=api, competition=competition)
    except Exception as e:  # noqa: BLE001
        if type(e) is requests.exceptions.HTTPError and e.response.status_code == ERROR_CODE_NOT_JOIN_COMPETITION:
            logger.error(f"You have not joined the competition '{competition}'.")
        else:
            logger.error(f"Fail to get submissions with error: {e}")


def generate_kaggle_kernel_metadata(user: str, competition: str) -> dict:
    """Generate kaggle kernel metadata to upload notebook to kaggle.

    Args:
        user (str): kaggle username
        competition (str): which competition this notebook will reference to

    Returns:
        dict: kaggle kernel metadata
    """
    kernel_name = f"{KG_SUBMISSION_SETTING.kernel_prefix}-{competition}"

    return {
        "id": f"{user}/{kernel_name}",
        "title": kernel_name,
        "code_file": "submission.ipynb",  # we hard coded the name, same with prepare_notebook function
        "language": "python",
        "kernel_type": "notebook",
        "is_private": KG_SUBMISSION_SETTING.is_private,
        "enable_gpu": KG_SUBMISSION_SETTING.enable_gpu,
        "enable_tpu": KG_SUBMISSION_SETTING.enable_tpu,
        "enable_internet": KG_SUBMISSION_SETTING.enable_internet,
        "dataset_sources": [],
        "competition_sources": [f"{competition}"],
        "kernel_sources": [],
        "model_sources": [],  # TODO: we can use this field if we want to upload our model
    }


# this is the minimal notebook template from kaggle notebook page,
# with this template, we can create a notebook without kernel error when uploading
kaggle_notebook_template = """
{
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "version": "3.6.4",
            "file_extension": ".py",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "name": "python",
            "mimetype": "text/x-python"
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
    ]
}
"""


def prepare_notebook(user: str, competition: str, workspace: Path, output_path: Path) -> None:
    """Prepare notebook from workspace for uploading to kaggle.

    NOTE: kaggle need to prepare a folder with kernel-metadata.json and xxx.ipynb to call api to upload

    Args:
        user (str): kaggle username used to construct the notebook url
        competition (str): which competition this notebook will reference to
        workspace (Path): where to get code content, and write to notebook
        output_path (Path): notebook output path
    """
    # NOTE: this function do not support competitions that contains more than one code files
    # generate metadata
    metadata = generate_kaggle_kernel_metadata(user=user, competition=competition)

    logger.info(f"Generated metadata: {metadata}")

    metadata_file = output_path / "kernel-metadata.json"

    # write the metadata
    with metadata_file.open("wt", encoding="utf-8") as fp:
        json.dump(metadata, fp)

    # create kaggle notebook from template without converting
    notebook: nbformat.NotebookNode = nbformat.reads(kaggle_notebook_template, as_version=nbformat.NO_CONVERT)

    eda_file = workspace / "EDA.md"

    # try to read and create a markdown cell for EDA.md
    if eda_file.exists():
        with eda_file.open("rt", encoding="utf-8") as fp:
            notebook.cells.append(nbformat.v4.new_markdown_cell(fp.read()))

    # read the main.py
    main_file = workspace / "main.py"

    if not main_file.exists():
        raise FileNotFoundError

    with main_file.open("rt", encoding="utf-8") as fp:
        main_code = fp.read()

    # update the main code to make it possible to run in kaggle
    # TODO: support other project structure
    # 1. change the input dir to kaggle input
    main_code = main_code.replace("./workspace_input", f"/kaggle/input/{competition}")

    # 2. patch the possible argparser code to support kaggle notebook starting parameter
    # we cannot just use replace here, as we need insert correct indent
    for code_line in main_code.split("\n"):
        if code_line.strip() == "parser = argparse.ArgumentParser()":
            # get the indent length
            indent_n = code_line.index("parser")
            # kaggle notebook will add additional arguments, we add them here to avoid error
            main_code = main_code.replace(
                "parser = argparse.ArgumentParser()",
                (
                    f"parser = argparse.ArgumentParser()\n"
                    f"{' ' * indent_n}parser.add_argument('-f', required=False)\n"
                    f"{' ' * indent_n}parser.add_argument('--HistoryManager.hist_file', required=False)\n"
                ),
            )

            break

    notebook.cells.append(nbformat.v4.new_code_cell(main_code))

    # an additional cell used to print traceback if there is any error
    notebook.cells.append(nbformat.v4.new_code_cell("%tb"))

    # NOTE: in kaggle, the default workspace is /kaggle/working,
    # so if our result files are writing into current folder, then we can use it to submit.
    # and we do not need to change the code

    nbformat.write(notebook, os.path.join(output_path, "submission.ipynb"))  # noqa: PTH118


def upload_notebook(api: KaggleApi, kernel_id: str, folder: Path) -> int | None:
    """Upload notebook to kaggle.

    Args:
        api (kaggle.api.KaggleApi): kaggle api
        kernel_id (str): id of the kernel, used to check status
        folder (Path): the folder to upload that contains kernel-metadata.json and submission.ipynb

    Return:
        int: the version number of uploaded kernel
    """
    # try to push the kernel to kaggle
    try:
        upload_resp: ApiSaveKernelResponse = api.kernels_push(folder=folder)

        # check upload status
        if upload_resp.error:
            logger.error(f"Upload notebook failed: {upload_resp.error}. respose object: {upload_resp}")

            return None
    except Exception as e:  # noqa: BLE001
        logger.error(f"Upload notebook failed: {e}")

        return None

    # wait until the kernel running is done
    try:
        while (kernel_status_resp := api.kernels_status(kernel_id)) is not None:
            logger.info(f"Kernel status: {kernel_status_resp.status}")

            if kernel_status_resp.status == KernelWorkerStatus.COMPLETE:
                break

            if kernel_status_resp.status in [
                KernelWorkerStatus.ERROR,
                KernelWorkerStatus.CANCEL_ACKNOWLEDGED,
                KernelWorkerStatus.CANCEL_REQUESTED,
            ]:
                logger.error(f"Kernel error: {kernel_status_resp.status}")

                return None

            # sleep more seconds to avoid rate limit, as notebook will cost more time to complete
            time.sleep(30)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Check notebook status failed: {e}")

        return None
    finally:
        logger.info(f"Refer to this link for details: {upload_resp.url}")

    return upload_resp.version_number


def download_kernel_output(api: KaggleApi, kernel_id: str, output_folder: Path) -> bool:
    """Download output files of a kernel.

    Args:
        api (kaggle.api.KaggleApi): kaggle api object
        kernel_id (str): kernel id to download
        output_folder (Path): output folder to save the output files

    Return:
        bool: whether download successfully
    """
    try:
        api.kernels_output(kernel=kernel_id, path=output_folder, force=True)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Download notebook output failed: {e}")

        return False

    return True


@contextmanager
def submit_notebook(api: KaggleApi, competition: str, workspace: Path) -> Generator[tuple[Path, str, int], None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = Path(tmp_dir)

        prepare_notebook(
            user=api.config_values["username"],
            competition=competition,
            workspace=workspace,
            output_path=kernel_path,
        )

        # read the metadata to get the kernel id
        metadata_json_file = kernel_path / "kernel-metadata.json"

        with metadata_json_file.open("rt", encoding="utf-8") as fp:
            kernel_metadata = json.load(fp)

        kernel_id = kernel_metadata["id"]

        kernel_version = upload_notebook(api=api, kernel_id=kernel_id, folder=kernel_path)

        if kernel_version is None:
            logger.error("Upload notebook failed")

            return

        yield (kernel_path, kernel_id, kernel_version)


def submit_notebook_output(
    api: KaggleApi,
    competition: str,
    workspace: Path,
    msg: str,
) -> None:
    """Submit code from workspace, download the output and then submit the output to competition.

    Args:
        api (kaggle.api.KaggleApi): kaggle api object
        competition (str): competition name
        workspace (Path): workspace path contains code to run
        msg (str): message to submit
    """
    with submit_notebook(api=api, competition=competition, workspace=workspace) as (kernel_path, kernel_id, _):
        if download_kernel_output(api=api, kernel_id=kernel_id, output_folder=kernel_path):
            # check if the submission file exist
            submission_file_path = kernel_path / KG_SUBMISSION_SETTING.submission_file

            if not submission_file_path.exists():
                logger.error(f"Submission file {submission_file_path} not found in kernel output")

                return

            submit_local_file(api=api, competition=competition, file=str(submission_file_path), msg=msg)


def submit_notebook_online(
    api: KaggleApi,
    competition: str,
    workspace: Path,
    msg: str,
) -> None:
    """Submit the output of online notebook in kaggle to competition.

    Args:
        api (kaggle.api.KaggleApi): kaggle api object
        competition (str): competition name
        workspace (Path): workspace path contains code to run
        msg (str): message to submit
    """
    with submit_notebook(api=api, competition=competition, workspace=workspace) as (_, kernel_id, kernel_version):
        submit_kernel_by_version(
            api=api,
            competition=competition,
            kernel_id=kernel_id,
            kernel_version=kernel_version,
            msg=msg,
        )


def submit_from_workspace(competition: str, workspace: Path | str, *, msg: str = "Message") -> None:
    """Submit the result of competition to kaggle from specified workspace.

    Args:
        competition (str): the competition name
        code_only (bool, optional): if True, submit a notebook, and download the output, then submit for non-code competitions.
    """
    api = KaggleApi()

    api.authenticate()

    competition_info = get_competition_detail(api=api, competition=competition)

    if competition_info is None:
        logger.warning(f"Cannot find competition: {competition}")

        return

    submission_file = KG_SUBMISSION_SETTING.submission_file

    if type(workspace) is str:
        workspace = Path(workspace)

    # if the competition is not code competition, and not force to use notebook, then submit local file
    if not KG_SUBMISSION_SETTING.force_submit_code and not competition_info.is_kernels_submissions_only:
        logger.info(f"Submitting {submission_file} to {competition}")

        file = workspace / submission_file  # type: ignore

        submit_local_file(api=api, competition=competition, file=file, msg=msg)
    elif not competition_info.is_kernels_submissions_only:
        logger.info(f"Submitting via notebook for {competition}")

        submit_notebook_output(api=api, competition=competition, workspace=workspace, msg=msg)  # type: ignore

        return
    else:
        logger.info(f"Submitting notebook for {competition}")

        submit_notebook_online(api=api, competition=competition, workspace=workspace, msg=msg)  # type: ignore

    # show remaining submit number
    # NOTE: if the competition is finished, the number will be 0
    remaining_num = get_submission_remaining(api=api, competition=competition)

    logger.info(f"Remaining submit number: {remaining_num}")


def submit_current_sota(competition: str) -> None:
    """Submit the sota result of competition from current trace path.

    Args:
        competition (str): which competition to submit
    """
    # we have to import this function here to avoid circular import issue
    from rdagent.log.conf import LOG_SETTINGS  # noqa: PLC0415
    from rdagent.log.ui.utils import get_sota_exp_stat  # noqa: PLC0415

    # check the trace_path
    sota, sota_loop_id, _, _ = get_sota_exp_stat(log_path=Path(LOG_SETTINGS.trace_path))

    logger.info(f"sota loop id: {sota_loop_id}")

    if sota is None:
        logger.warning("Cannot find sota experiment, skip submitting.")

        return

    if sota.experiment_workspace is None:
        logger.warning("Fail to get sota output, workspace is None.")

        return

    worspace = sota.experiment_workspace.workspace_path

    logger.info(f"Current sota workspace: {worspace}")

    # do submit
    submit_from_workspace(
        competition=competition,
        workspace=worspace,
        msg=f"SOTA at {datetime.now()}, loop: {sota_loop_id}, file: {KG_SUBMISSION_SETTING.submission_file}",  # noqa: DTZ005
    )


if __name__ == "__main__":
    fire.Fire({"sota": submit_current_sota, "workspace": submit_from_workspace})
