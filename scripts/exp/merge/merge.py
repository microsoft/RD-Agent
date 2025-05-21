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
from pathlib import Path

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.timer import RD_Agent_TIMER_wrapper


def get_session(path):
    with open(path, "rb") as f:
        session = pickle.load(f)
    return session


# for c in LITE:
for c in AIDEMEDAL:
    cand = []
    # for p in "/Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/tight-sponge/combined_logs", "/Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/neat-sunbird/combined_logs":
    # for p in "/data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/alert-monster/combined_logs;/data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/stirring-akita/combined_logs".split( ";"):
    for (
        p
    ) in "/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/deciding-cod/combined_logs;/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/exotic-frog/combined_logs".split(
        ";"
    ):
        p = Path(p) / f"{c}.1"
        # Find the ID_record in the maximum loop using a concise approach
        # Use glob to find the highest loop number and ensure 4_record exists

        session_path = p / "__session__"

        loop_dirs = session_path.glob("*/4_record")

        max_loop_record_path = max(loop_dirs, key=lambda x: int(x.parent.name))
        if max_loop_record_path:
            print(f"Max record path for {c}: {max_loop_record_path}")
            cand.append(max_loop_record_path)
        else:
            print(f"No record found for {c}")

    session_l = [get_session(p) for p in cand]

    assert all(s.trace.scen.metric_direction for s in session_l) or all(
        not s.trace.scen.metric_direction for s in session_l
    )

    # direction = session_l.trace[0].scen.metric_direction

    trace = session_l[0].trace

    orig_len = len(trace.hist)

    trace.hist.extend(session_l[1].trace.hist)
    trace.dag_parent.extend(tuple(p + orig_len for p in tp) for tp in session_l[1].trace.dag_parent)

    leaves = trace.get_leaves()
    print(c, leaves)
    assert len(leaves) == 2

    # session_l[0].trace.hist
    # session_l[0].trace.dag_parent

    new_session_dir = Path("sessions")

    new_session_dir.mkdir(exist_ok=True, parents=True)

    session_l[0].timer = RD_Agent_TIMER_wrapper.timer
    session_l[0].step_idx = 0
    session_l[0].loop_idx = len(trace.hist)
    with (new_session_dir / f"{c}").open("wb") as f:
        pickle.dump(session_l[0], f)
