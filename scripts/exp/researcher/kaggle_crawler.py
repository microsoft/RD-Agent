# %%
import bisect
import json
import shutil
import subprocess
import time
import zipfile
from itertools import chain
from pathlib import Path
import logging

import nbformat
from jinja2 import Environment, StrictUndefined
from rich import print
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.exception import KaggleError
from rdagent.core.prompts import Prompts
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.debug.data import create_debug_data
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import MLEBDockerEnv

from tqdm import tqdm
from bs4 import BeautifulSoup
# %%
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--headless")

service = Service("/usr/local/bin/chromedriver")


def solution_to_feature(inputs) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_feature"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["solution_to_feature"]["user"])
        .render(inputs=inputs)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def crawl_discussions(
        competition: str, local_data_path: str, wait: float = 3.0, force: bool = False, num: int = 10
    ) -> dict[dict]: 
    output_path = Path(f"{local_data_path}/{competition}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        logger.info(f"Found {output_path}, loading from it.")
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)
        
    driver = webdriver.Chrome(options=options, service=service)

    discussion_urls = []
    page = 1
    while len(discussion_urls) < num: 
        if page == 1: 
            discussion_url = f"https://www.kaggle.com/competitions/{competition}/discussion?sort=hotness"
        else: 
            discussion_url = f"https://www.kaggle.com/competitions/{competition}/discussion?sort=hotness&page={page}"
        driver.get(discussion_url)
        time.sleep(wait)  # wait for the page to load

        # extract links to discussion pages
        discussion_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/discussion/']")

        # only crawl topics under 'All other topics'
        try: 
            all_other_topics = driver.find_elements(By.XPATH, "//h3[contains(text(), 'All other topics')]")[0]
        except: 
            all_other_topics = None

        for link in discussion_links:
            if all_other_topics:  
                if link.location['y'] < all_other_topics.location['y']: # above 'All other topics'
                    continue
            url = link.get_attribute("href") 
            if "#" not in url.split("/")[-1]: # filter out comment pages
                discussion_urls.append(url)
        
        page += 1

    def extract_discussion_content(url):
        driver.get(url)
        time.sleep(5)
        title = driver.find_element(By.CSS_SELECTOR, "h3").text
        body = driver.find_element(By.XPATH, "//h3/following-sibling::div[1]")

        def remove_attributes(html):
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(True):  # find all tags
                tag.attrs = {}  # remove all attributes
            return str(soup)

        return {
            "title": title,
            "body_text": body.text,
            "body_html": remove_attributes(body.get_attribute("innerHTML")),
        }
    
    discussions = []
    for url in tqdm(discussion_urls[:num], desc=f"Crawling discussion from {competition}"):
        discussions.append(extract_discussion_content(url))
    driver.quit()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(discussions, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved {len(discussions)} discussions to {output_path}.")
    return discussions


def discussion_to_knowledge(discussion_text: str) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["gen_knowledge_from_discussion"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["gen_knowledge_from_discussion"]["user"])
        .render(discussion=discussion_text)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def naive_solution_gen(competition: str) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["naive_solution_gen"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["naive_solution_gen"]["user"])
        .render(competition=competition)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def convert_discussion_to_text(
    competition: str, local_path: str = f"{KAGGLE_IMPLEMENT_SETTING.local_data_path}"
) -> None: 

    dir = Path(local_path)
    for json_file in dir.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, part in enumerate(data):
                title = part.get('title', '')
                body_content = part.get('body', '')
                text = f"Title: {title}\nBody: {body_content}"
                text = discussion_to_knowledge(text)
                output_file = dir / f"{json_file.stem}_part{i+1}.txt"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(text)


def crawl_descriptions(
    competition: str, local_data_path: str, wait: float = 3.0, force: bool = False
) -> dict[str, str]:
    if (fp := Path(f"{local_data_path}/{competition}/description.md")).exists() and not force:
        logger.info(f"Found {competition}/description.md, loading from it.")
        return fp.read_text()

    if (fp := Path(f"{local_data_path}/{competition}.json")).exists() and not force:
        logger.info(f"Found {competition}.json, loading from local file.")
        with fp.open("r") as f:
            return json.load(f)

    driver = webdriver.Chrome(options=options, service=service)
    overview_url = f"https://www.kaggle.com/competitions/{competition}/overview"
    driver.get(overview_url)
    time.sleep(wait)
    site_body = driver.find_element(By.ID, "site-content")
    descriptions = {}

    # Get the subtitles
    elements = site_body.find_elements(By.CSS_SELECTOR, f"a[href^='/competitions/{competition}/overview/']")
    subtitles = []
    for e in elements:
        inner_text = ""
        for child in e.find_elements(By.XPATH, ".//*"):
            inner_text += child.get_attribute("innerHTML").strip()
        subtitles.append(inner_text)

    def kaggle_description_css_selectors() -> tuple[str, str]:
        # Get the class name of the main contents
        ab_elm = site_body.find_element(By.ID, "abstract")
        others_elm = ab_elm.find_element(By.XPATH, "../*[2]")
        first_elm = others_elm.find_element(By.XPATH, "./*[1]")
        first_content_elm = first_elm.find_element(By.XPATH, "./*[1]/*[2]")
        selector_elm = first_content_elm.find_element(By.XPATH, "./*[1]/*[1]")
        main_class = selector_elm.get_attribute("class").split()[-1]

        # Get the class name of the citation
        citation_elm = site_body.find_element(By.ID, "citation")
        citation_content_elm = citation_elm.find_element(By.XPATH, "./*[1]/*[2]/*[1]/*[1]")
        citation_class = citation_content_elm.get_attribute("class").split()[-1]

        return main_class, citation_class

    main_class, citation_class = kaggle_description_css_selectors()

    # Get main contents
    contents = []
    elements = site_body.find_elements(By.CSS_SELECTOR, f".{main_class}")
    for e in elements:
        content = e.get_attribute("innerHTML")
        contents.append(content)

    assert len(subtitles) == len(contents) + 1 and subtitles[-1] == "Citation"
    for i in range(len(subtitles) - 1):
        descriptions[subtitles[i]] = contents[i]

    # Get the citation
    element = site_body.find_element(By.CSS_SELECTOR, f".{citation_class}")
    citation = element.get_attribute("innerHTML")
    descriptions[subtitles[-1]] = citation

    data_url = f"https://www.kaggle.com/competitions/{competition}/data"
    driver.get(data_url)
    time.sleep(wait)
    data_element = driver.find_element(By.CSS_SELECTOR, f".{main_class}")
    descriptions["Data Description"] = data_element.get_attribute("innerHTML")

    driver.quit()
    with open(f"{local_data_path}/{competition}.json", "w") as f:
        json.dump(descriptions, f)
    return descriptions


def download_data(competition: str, settings: ExtendedBaseSettings = KAGGLE_IMPLEMENT_SETTING) -> None:
    local_path = settings.local_data_path
    if settings.if_using_mle_data:
        zipfile_path = f"{local_path}/zip_files"
        zip_competition_path = Path(zipfile_path) / competition

        mleb_env = MLEBDockerEnv()
        mleb_env.prepare()
        if not zip_competition_path.exists():
            (Path(zipfile_path)).mkdir(parents=True, exist_ok=True)
            mleb_env.run(
                f"mlebench prepare -c {competition} --data-dir ./zip_files",
                local_path=local_path,
                running_extra_volume={str(Path("~/.kaggle").expanduser().absolute()): "/root/.kaggle"},
            )

        if not (Path(local_path) / competition).exists() or list((Path(local_path) / competition).iterdir()) == []:
            (Path(local_path) / competition).mkdir(parents=True, exist_ok=True)

            mleb_env.run(
                f"/bin/sh -c 'cp -r ./zip_files/{competition}/prepared/public/* ./{competition}'", local_path=local_path
            )
            mleb_env.run(
                f'/bin/sh -c \'for zip_file in ./{competition}/*.zip; do dir_name="${{zip_file%.zip}}"; mkdir -p "$dir_name"; unzip -o "$zip_file" -d "$dir_name"; done\'',
                local_path=local_path,
            )
            # NOTE:
            # Patching:  due to mle has special renaming mechanism for different competition;
            # We have to switch the schema back to a uniform one;
            if competition in {"new-york-city-taxi-fare-prediction"}:
                cpath = Path(local_path) / f"{competition}"
                labels_path = cpath / "labels.csv"
                train_path = cpath / "train.csv"
                if labels_path.exists():
                    shutil.copy(labels_path, train_path)
                else:
                    logger.error(f"labels.csv not found in {cpath}")
                    raise FileNotFoundError(f"{labels_path} does not exist")
    else:
        zipfile_path = f"{local_path}/zip_files"
        if not Path(f"{zipfile_path}/{competition}.zip").exists():
            try:
                subprocess.run(
                    ["kaggle", "competitions", "download", "-c", competition, "-p", zipfile_path],
                    check=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Download failed: {e}, stderr: {e.stderr}, stdout: {e.stdout}")
                raise KaggleError(f"Download failed: {e}, stderr: {e.stderr}, stdout: {e.stdout}")

            # unzip data
            unzip_path = f"{local_path}/{competition}"
            if not Path(unzip_path).exists():
                unzip_data(unzip_file_path=f"{zipfile_path}/{competition}.zip", unzip_target_path=unzip_path)
                for sub_zip_file in Path(unzip_path).rglob("*.zip"):
                    unzip_data(sub_zip_file, unzip_target_path=unzip_path)

    # sample data
    if not Path(f"{local_path}/sample/{competition}").exists():
        create_debug_data(competition, dataset_path=local_path)


def unzip_data(unzip_file_path: str, unzip_target_path: str) -> None:
    with zipfile.ZipFile(unzip_file_path, "r") as zip_ref:
        zip_ref.extractall(unzip_target_path)


def leaderboard_scores(competition: str) -> list[float]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    ll = api.competition_leaderboard_view(competition)
    return [float(x.score) for x in ll]


def score_rank(competition: str, score: float) -> tuple[int, float]:
    """
    Return
    ------
    rank: int
    rank_percent: float
    """
    scores = leaderboard_scores(competition)
    if scores[0] < scores[-1]:  # Ascending order
        rank = bisect.bisect_right(scores, score)
    else:  # Descending order
        scores = scores[::-1]  # Reverse the list to use bisect
        rank = len(scores) - bisect.bisect_right(scores, score)

    rank = rank + 1
    rank_percent = rank / len(scores) * 100

    return rank, rank_percent


def download_notebooks(
    competition: str, local_path: str = f"{KAGGLE_IMPLEMENT_SETTING.local_data_path}/notebooks", num: int = 15
) -> None:
    data_path = Path(f"{local_path}/{competition}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # judge the sort_by
    ll = api.competition_leaderboard_view(competition)
    score_diff = float(ll[0].score) - float(ll[-1].score)
    if score_diff > 0:
        sort_by = "scoreDescending"
    else:
        sort_by = "scoreAscending"

    # download notebooks
    nl = api.kernels_list(competition=competition, sort_by=sort_by, page=1, page_size=num)
    for nb in nl:
        author = nb.ref.split("/")[0]
        api.kernels_pull(nb.ref, path=data_path / author)
    print(f"Downloaded {len(nl)} notebooks for {competition}. ([red]{sort_by}[/red])")


def notebook_to_knowledge(notebook_text: str) -> str:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["gen_knowledge_from_code_mini_case"]["system"])
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["gen_knowledge_from_code_mini_case"]["user"])
        .render(notebook=notebook_text)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def knowledge_base_generator(competition_text: str, notebook_text: str) -> dict:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["notebook_to_knowledge"]["system"])  
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["notebook_to_knowledge"]["user"]) 
        .render(competition=competition_text, notebook=notebook_text)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    
    logging.info("Raw Response: %s", response)

    if response is None or response.strip() == "":
        logging.warning("Received an empty or None response. Skipping this competition.")
        return {"Error": "Received an empty response."}

    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    response_dict = json.loads(cleaned_response)
    return response_dict


def idea_merger(old_idea: str, new_idea: str) -> dict:
    prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    
    sys_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["idea_merg"]["system"])  
        .render()
    )

    user_prompt = (
        Environment(undefined=StrictUndefined)
        .from_string(prompt_dict["idea_merg"]["user"]) 
        .render(old_idea=old_idea, new_idea=new_idea)
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    
    logging.info("Raw Response: %s", response)

    if response is None or response.strip() == "":
        logging.warning("Received an empty or None response. Skipping this competition.")
        return {"Error": "Received an empty response."}

    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    response_dict = json.loads(cleaned_response)
    return response_dict


def convert_notebooks_to_text(
    competition: str, local_path: str = f"{KAGGLE_IMPLEMENT_SETTING.local_data_path}/notebooks"
) -> None:
    data_path = Path(f"{local_path}/{competition}")
    converted_num = 0

    # convert ipynb and irnb files
    for nb_path in chain(data_path.glob("**/*.ipynb"), data_path.glob("**/*.irnb")):
        with nb_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        text = []
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                text.append(f"```markdown\n{cell.source}```")
            elif cell.cell_type == "code":
                text.append(f"```code\n{cell.source}```")
        text = "\n\n".join(text)

        text = notebook_to_knowledge(text)

        text_path = nb_path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        converted_num += 1

    # convert py files
    for py_path in data_path.glob("**/*.py"):
        with py_path.open("r", encoding="utf-8") as f:
            text = f"```code\n{f.read()}```"

        text = notebook_to_knowledge(text)

        text_path = py_path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        converted_num += 1

    print(f"Converted {converted_num} notebooks to text files.")


def collect_knowledge_texts(local_path: str = KAGGLE_IMPLEMENT_SETTING.local_data_path) -> dict[str, list[str]]:
    """
    {
        "competition1": [
            "knowledge_text1",
            "knowledge_text2",
            ...
        ],
        “competition2”: [
            "knowledge_text1",
            "knowledge_text2",
            ...
        ],
        ...
    }
    """
    notebooks_dir = Path(local_path) / "notebooks"

    competition_knowledge_texts_dict = {}
    for competition_dir in notebooks_dir.iterdir():
        knowledge_texts = []
        for text_path in competition_dir.glob("**/*.txt"):
            text = text_path.read_text(encoding="utf-8")
            knowledge_texts.append(text)

        competition_knowledge_texts_dict[competition_dir.name] = knowledge_texts

    return competition_knowledge_texts_dict


# %%
if __name__ == "__main__":
    mini_case_cs = [
        "feedback-prize-english-language-learning",
        "playground-series-s3e11",
        "playground-series-s3e14",
        "spaceship-titanic",
        "playground-series-s3e18",
        "playground-series-s3e16",
        "playground-series-s3e9",
        "playground-series-s3e25",
        "playground-series-s3e26",
        "playground-series-s3e24",
        "playground-series-s3e23",
    ]

    other_cs = [
        "amp-parkinsons-disease-progression-prediction",
        "arc-prize-2024",
        "ariel-data-challenge-2024",
        "child-mind-institute-detect-sleep-states",
        "connectx",
        "contradictory-my-dear-watson",
        "digit-recognizer",
        "fathomnet-out-of-sample-detection",
        "forest-cover-type-prediction",
        "gan-getting-started",
        "google-research-identify-contrails-reduce-global-warming",
        "house-prices-advanced-regression-techniques",
        "isic-2024-challenge",
        "leash-BELKA",
        "llm-20-questions",
        "nlp-getting-started",
        "playground-series-s4e1",
        "playground-series-s4e2",
        "playground-series-s4e3",
        "playground-series-s4e4",
        "playground-series-s4e5",
        "playground-series-s4e6",
        "playground-series-s4e7",
        "playground-series-s4e8",
        "rsna-2024-lumbar-spine-degenerative-classification",
        "sf-crime",
        "store-sales-time-series-forecasting",
        "titanic",
        "tpu-getting-started",
        # scenario competition
        "covid19-global-forecasting-week-1",
        "statoil-iceberg-classifier-challenge",
        "optiver-realized-volatility-prediction",
        "facebook-v-predicting-check-ins",
    ]

    # all_cs = mini_case_cs + other_cs
    # for c in all_cs:
    #     convert_notebooks_to_text(c)
    # exit()
    # from kaggle.api.kaggle_api_extended import KaggleApi

    # api = KaggleApi()
    # api.authenticate()
    # cs = api.competitions_list()
    # for c in cs:
    #     name = c.ref.split("/")[-1]
    #     crawl_descriptions(name)
    res = leaderboard_scores(competition="playground-series-s4e8")
    rank, rank_percent = score_rank(competition="playground-series-s4e8", score=0.9832)
    print(rank, rank_percent)
# %%