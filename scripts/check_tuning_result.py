from rdagent.log.storage import FileStorage
from pathlib import Path
from collections import defaultdict
import pandas as pd
from io import StringIO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Your Kaggle API key is readable by other users")

from rdagent.log.utils import extract_loopid_func_name, extract_evoid
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction
import os

log_path_template = "/data/userdata/v-xuminrui/JobAndExp/amlt_project/amlt/{exp}/combined_logs/{competition}.1"

def cal_evolving_score(runner_feedback):
    if runner_feedback.score is None:
        return None
    score_df = pd.read_csv(StringIO(runner_feedback.score), sep='\s+')
    try:
        score = score_df.loc['ensemble'].iloc[0]
    except:
        score = score_df.loc[score_df.iloc[:, 0] == 'ensemble', score_df.columns[1]].values[0]
    return score

def get_evolving_feedbacks(exp, competition):
    log_path = Path(log_path_template.format(exp=exp, competition=competition))
    if not log_path.exists():
        return None
    evolving_feedbacks = defaultdict(dict)
    
    for msg in FileStorage(log_path).iter_msg(tag="running"):
        loop_id, fn = extract_loopid_func_name(msg.tag)
        evo_id = extract_evoid(msg.tag)
        if evo_id is not None and "evolving feedback" in msg.tag:
            runner_feedback = msg.content[0]
            runner_score = cal_evolving_score(runner_feedback)
            evolving_feedbacks[int(loop_id)][int(evo_id)] = {
                "runner_feedback": runner_feedback,
                "runner_score": runner_score,
            }
        
    return evolving_feedbacks

def compare_two_scores(base_score, cur_score, competition):
    metric_direction = get_metric_direction(competition)
    if base_score is None:
        if cur_score is None:
            return False
        else:
            return True
    else:
        if cur_score is None: # no improve
            return False
        else:
            score_diff = cur_score - base_score
            if (score_diff > 0 and metric_direction) or (score_diff < 0 and not metric_direction):
                return True
            
def analyze_evolving_feedbacks(evolving_feedbacks, competition):
    # Statistics
    count_vs_base = imp_vs_base = 0  # improvement vs base
    count_vs_prev = imp_vs_prev = 0  # improvement vs previous
    count_vs_sota = imp_vs_sota = 0  # improvement vs sota

    sota_score = None
    loops = sorted(evolving_feedbacks.keys())
    for loop_id in loops:
        runner_exp = evolving_feedbacks[loop_id]
        if runner_exp:
            evo_loops = sorted(evolving_feedbacks[loop_id].keys())

            base_score = prev_score = runner_exp[0]['runner_score']
            if sota_score is None and base_score is not None:
                sota_score = base_score

            for evo_id in evo_loops[1:]:
                cur_score = runner_exp[evo_id]['runner_score']
                if sota_score is None and cur_score is not None:
                    sota_score = cur_score

                # Evaluation 1: improvement over the base score
                if base_score is not None:
                    count_vs_base += 1
                    if compare_two_scores(base_score, cur_score, competition):
                        imp_vs_base += 1

                # Evaluation 2: improvement over the previous score
                if prev_score is not None:
                    count_vs_prev += 1
                    if compare_two_scores(prev_score, cur_score, competition):
                        imp_vs_prev += 1
                        prev_score = cur_score

                # Evaluation 3: improvement over the sota score
                if sota_score is not None:
                    count_vs_sota += 1
                    if compare_two_scores(sota_score, cur_score, competition):
                        imp_vs_sota += 1
                        sota_score = cur_score
                
    return {
        "base_count": count_vs_base,
        "base_imp": imp_vs_base,
        "prev_count": count_vs_prev,
        "prev_imp": imp_vs_prev,
        "sota_count": count_vs_sota,
        "sota_imp": imp_vs_sota,
    }

def save_to_md(exp, competitions, results):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{exp}_results.md", "w") as f:
        f.write(f"# Results for {exp}\n\n")
        f.write("| Competition | Base Ratio | Prev Ratio | SOTA Ratio |\n")
        f.write("|-------------|------------|------------|------------|\n")
        for competition in competitions:
            result = results.get(competition, {})
            base_ratio = (
                f"{result['base_imp']}/{result['base_count']} ({result['base_imp']/result['base_count']:.2f})"
                if result.get('base_count', 0) else "N/A"
            )
            prev_ratio = (
                f"{result['prev_imp']}/{result['prev_count']} ({result['prev_imp']/result['prev_count']:.2f})"
                if result.get('prev_count', 0) else "N/A"
            )
            sota_ratio = (
                f"{result['sota_imp']}/{result['sota_count']} ({result['sota_imp']/result['sota_count']:.2f})"
                if result.get('sota_count', 0) else "N/A"
            )
            f.write(f"| {competition} | {base_ratio} | {prev_ratio} | {sota_ratio} |\n")

def main(experiments: list, competitions: list):
    for exp in experiments:
        results = {}
        for competition in competitions:
            evolving_feedbacks = get_evolving_feedbacks(exp, competition)
            if evolving_feedbacks:
                result = analyze_evolving_feedbacks(evolving_feedbacks, competition)
                results[competition] = result
        save_to_md(exp, competitions, results)

if __name__ == "__main__":

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

    RDMEDAL = [
        "whale-categorization-playground",
        "seti-breakthrough-listen",
        "learning-agency-lab-automated-essay-scoring-2",
    ]

    NEWMEDAL = [
        "iwildcam-2020-fgvc7",
        "herbarium-2022-fgvc9",
        "freesound-audio-tagging-2019",
    ]

    experiments = ["diverse-mammoth", "liberal-swan", "stable-racer", "ready-haddock", "moved-coral"]
    competitions = AIDEMEDAL + RDMEDAL + NEWMEDAL
    main(experiments, competitions)