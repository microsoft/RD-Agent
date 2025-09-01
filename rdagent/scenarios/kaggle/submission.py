import time
import requests

from typing import List, Optional
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi, SubmissionSortBy, SubmissionStatus
from kagglesdk.competitions.types.competition_api_service import (
    ApiSubmission,
    ApiCompetition,
)
from rdagent.log import rdagent_logger as logger


def get_rest_submit_num(competition: str) -> int:
    """Get the remaining submit number.

    NOTE: this function will try to avoid any exception raised, so if you have not joined the competition, or any other exceptions, will return 0
    """
    api = KaggleApi()

    api.authenticate()

    # kaggle use utc time to count
    now = datetime.now(timezone.utc)
    today = now.date()

    try:
        submissions_resp = api.competition_submissions(
            competition=competition,
            # the api use date to sort the result by default, so latest submission is in last page
            page_token=-1,
        )
    except Exception as e:
        # 403 if we have not joined the competition
        if type(e) == requests.exceptions.HTTPError and e.response.status_code == 403:
            logger.error(f"You have not joined the competition '{competition}'.")
        else:
            logger.error(f"Fail to get submissions with error: {e}")

        return 0

    if submissions_resp is None:
        logger.error(f"Fail to get submissions, response is None.")
        return 0

    # only completed submission will consume today's limitation
    completed_submissions: List[ApiSubmission] = []
    # we will give a warning, if there is any pendding submissions, as we do not if it should be counted
    pendding_submissions: List[ApiSubmission] = []

    for submission in submissions_resp:
        if submission is None or submission.date.date() != today:
            continue

        if submission.status == SubmissionStatus.COMPLETE:
            completed_submissions.append(submission)
        elif submission.status == SubmissionStatus.PENDING:
            pendding_submissions.append(submission)

    # TODO: may be it is better to treat pendding competitions as completed?
    if len(pendding_submissions) > 0:
        logger.warning(
            f"Completition has {len(pendding_submissions)} pendding submission now, the remaining submit number may be not correct."
        )

    # search competition by name
    def _get_competition(max_pages: int = 1) -> Optional[ApiCompetition]:
        # list function only return in-progress competitions,
        # and we will use search parameter to filter competition name,
        # and default page_size is 20, so using 1 page is enough by default
        for page in range(1, max_pages + 1):
            try:
                search_result = api.competitions_list(search=competition, page=page)
            except Exception as e:
                logger.error(f"Fail to get competition list, with exception: {e}")

                return None

            if search_result is not None:
                for comp in search_result:
                    if comp is not None and comp.ref.rsplit("/", 1)[-1] == competition:
                        return comp

    competition_detail = _get_competition()

    if competition_detail is None:
        logger.error(f"Fail to get the competition, make sure it is in progress now.")

        return 0

    return competition_detail.max_daily_submissions - len(completed_submissions)


def submit_csv(
    competition: str, submission_file: str, msg: str = "Message", wait: bool = False, timeout: int = 120
) -> bool:
    """Submit csv file to competition."""
    api = KaggleApi()

    api.authenticate()

    # submit first
    try:
        resp = api.competition_submit(file_name=submission_file, message=msg, competition=competition)
    except Exception as e:
        if type(e) == requests.exceptions.HTTPError and e.response.status_code == 403:
            logger.error(f"You have not joined the competition '{competition}'.")
        else:
            logger.error(f"Fail to get submissions with error: {e}")

        return False

    # if upload failed, will return a string message, or it will raise exception if there is http response code >= 400
    if type(resp) == str:
        return False

    is_success = False

    # the submission request is done without error
    if wait:
        start = datetime.now()
        while (datetime.now() - start).seconds <= timeout:
            submission_resp = api.competition_submissions(
                # the api can sort the submissions by date (new -> old), so we use the first item as first page
                competition=competition,
                sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
                page_token=0,
            )

            logger.info(f"Current submissions: {submission_resp}")

            if (
                submission_resp is None
                or submission_resp[0] is None
                or submission_resp[0].status == SubmissionStatus.PENDING
            ):
                # wait for 2 seconds to get latest result
                time.sleep(2)

                continue

            submission = submission_resp[0]

            if submission.status == SubmissionStatus.ERROR:
                logger.error(f"The submission validation failed, with message: {submission.error_description}")
                is_success = False
            else:
                logger.info(f"Current submission state [@{submission.date}] is: {submission.status}")

            break

    return is_success


if __name__ == "__main__":
    csv_path = "data/playground-series-s5e9/sample_submission.csv"
    competition_name = "playground-series-s5e9"

    # submit and wait for compelete
    submit_csv(competition_name, csv_path, wait=True)

    # check the remaining submit number for today
    remaining_submit_num = get_rest_submit_num(competition_name)

    print(f"Remaining submit number: {remaining_submit_num}")
