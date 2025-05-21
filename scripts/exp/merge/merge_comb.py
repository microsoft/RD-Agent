from loguru import logger

LITE = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]

AIDEMEDAL = [
    "plant-pathology-2021-fgvc8",
    "dogs-vs-cats-redux-kernels-edition",
    "tabular-playground-series-dec-2021",
    "histopathologic-cancer-detection",
    "predict-volcanic-eruptions-ingv-oe",
    "nomad2018-predict-transparent-conductors",
    "text-normalization-challenge-english-language",
    "plant-pathology-2020-fgvc7",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "google-quest-challenge",
    "hotel-id-2021-fgvc8",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "herbarium-2020-fgvc7",
    "stanford-covid-vaccine",
    "aptos2019-blindness-detection",
    "spooky-author-identification",
    "random-acts-of-pizza",
    "jigsaw-toxic-comment-classification-challenge",
    "text-normalization-challenge-russian-language",
    "aerial-cactus-identification",
    "leaf-classification",
    "mlsp-2013-birds",
    "h-and-m-personalized-fashion-recommendations",
    "iwildcam-2019-fgvc6",
    "us-patent-phrase-to-phrase-matching",
    "tweet-sentiment-extraction",
    "kuzushiji-recognition",
    "herbarium-2021-fgvc8",
    "cassava-leaf-disease-classification",
    "3d-object-detection-for-autonomous-vehicles",
    "inaturalist-2019-fgvc6",
]

import pickle
from itertools import combinations
from pathlib import Path

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.timer import RD_Agent_TIMER_wrapper


def get_session(path):
    with open(path, "rb") as f:
        session = pickle.load(f)
    return session


exp_list = [
    "deciding-cod",
    "exotic-frog",  # 12h version
    "civil-reindeer",
    "selected-worm",  # o3-4.1 | 12h version
    "prepared-salmon",  # o1-4.1
]


def get_folder(exp_name):
    return Path("/home/xiaoyang/repos/JobAndExp/amlt_project/amlt") / exp_name / "combined_logs"


def loop_duration(loop_entries):
    """
    Calculate the total duration of a loop given its list of LoopTrace entries.
    Returns the duration in seconds.
    """
    if not loop_entries:
        return 0.0
    return (loop_entries[-1].end - loop_entries[0].start).total_seconds()


def truncate_session(exp_name, session):
    """
    Truncate the session trace for certain experiments.
    For experiments 'civil-reindeer', 'selected-worm', and 'prepared-salmon', remove the final loops
    that take up approximately half of the total session time.
    """
    if exp_name in ("deciding-cod", "exotic-frog"):
        # 12h trace does not need to be truncated
        return
    if exp_name in ("civil-reindeer", "selected-worm", "prepared-salmon"):
        # Calculate total session time based on the earliest and latest loop entries
        start_idx = min(session.loop_trace.keys())
        end_idx = max(session.loop_trace.keys())
        total_time = (session.loop_trace[end_idx][-1].end - session.loop_trace[start_idx][0].start).total_seconds()
        target = total_time / 2.0
        # Accumulate durations in reverse order using the loop_duration function
        cumulative, m = 0.0, 0
        for loop_idx in sorted(session.loop_trace.keys(), reverse=True):
            duration = loop_duration(session.loop_trace[loop_idx])
            cumulative += duration
            m += 1
            if cumulative >= target:
                break
        # Calculate the cutoff index and truncate the session trace accordingly
        trace_loop_num = len(session.loop_trace) - m
        logger.info(f"From {len(session.trace.hist)} to {trace_loop_num} loops for {exp_name}")
        session.trace.hist = session.trace.hist[:trace_loop_num]
        session.trace.dag_parent = session.trace.dag_parent[:trace_loop_num]


for c in AIDEMEDAL:
    for exp1, exp2 in combinations(exp_list, 2):
        session_l = []
        for exp_name in [exp1, exp2]:
            folder = get_folder(exp_name)
            p = folder / f"{c}.1"
            session_path = p / "__session__"
            # Ensure we turn the glob result into a list so max() is safe
            loop_dirs = list(session_path.glob("*/4_record"))
            if loop_dirs:
                max_loop_record_path = max(loop_dirs, key=lambda x: int(x.parent.name))
                print(f"Max record path for {c} from {folder}: {max_loop_record_path}")
                session = get_session(max_loop_record_path)
                truncate_session(exp_name, session)
                session_l.append(session)
            else:
                print(f"No record found for {c} in {folder}")
        if len(session_l) != 2:
            print(f"Skipping {c} for experiment pair {exp1}, {exp2}")
            continue

        assert all(s.trace.scen.metric_direction for s in session_l) or all(
            not s.trace.scen.metric_direction for s in session_l
        )

        trace = session_l[0].trace
        orig_len = len(trace.hist)
        trace.hist.extend(session_l[1].trace.hist)
        trace.dag_parent.extend(tuple(p + orig_len for p in tp) for tp in session_l[1].trace.dag_parent)
        leaves = trace.get_leaves()
        print(c, exp1, exp2, leaves)
        # assert len(leaves) == 2
        # Check if exactly two leaves are present; if not, log a warning and skip this candidate's merge.
        if len(leaves) != 2:
            logger.warning(
                f"The number of leaves is not 2 for candidate '{c}' with experiments {exp1} and {exp2}: {leaves}"
            )
            continue

        # Update the timer reference and reset indices for the merged merged session
        session_l[0].timer = RD_Agent_TIMER_wrapper.timer
        session_l[0].step_idx = 0
        session_l[0].loop_idx = len(trace.hist)

        new_session_dir = Path("sessions")
        # Create directory structure: sessions/<exp1>_<exp2>/<c>/
        exp_pair_dir = new_session_dir / f"{exp1}_{exp2}"
        exp_pair_dir.mkdir(exist_ok=True, parents=True)

        # Save the merged session as session.pkl in the candidate's subdirectory
        with (exp_pair_dir / f"{c}").open("wb") as f:
            pickle.dump(session_l[0], f)
