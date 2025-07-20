import asyncio
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from tqdm import tqdm

from rdagent.utils.agent.tpl import T

Provider = "gpt-4.1"
api_token = "sk-1234"
base_url = "http://10.150.240.117:38888"

MAX_RETRIES = 3
RETRY_DELAY = 5
RETRY_BACKOFF = 2


def download_kaggle_notebook(notebook_link: str, save_path: str | Path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    kernel_owner, kernel_slug = notebook_link.split("/")[-2], notebook_link.split("/")[-1]
    try:
        api.kernels_pull(f"{kernel_owner}/{kernel_slug}", path=save_path, metadata=False)
        print(f"✅ Successfully download {notebook_link} to {save_path}.")
    except Exception as e:
        print(f"❌ Failed to download notebook {notebook_link}: {e}")


class DiscussionPageInfo(BaseModel):
    title: str = Field(..., description="The discussion title.")
    content: str = Field(
        ...,
        description="The main methodological content of the discussion, structured in Markdown. Focus exclusively on technical approaches, methods, experiments, results, and reasoning related to the competition or problem. Ignore introductions, personal stories, or unrelated context.",
    )
    comments: List[Dict[str, Any]] = Field(
        ...,
        description="The ordered comments, each including the author and only the methodological content of the comment, structured in Markdown. Exclude any comments that are off-topic, congratulatory, social, or unrelated to methodology.",
    )
    notebook_link: str = Field(
        "",
        description="The link of the code notebook of this discussion, if any. The format should be https://www.kaggle.com/code/{username}/{notebook-name}. If there is no code link, it should be an empty string.",
    )


def create_crawler_config(instruction_name: str, schema: BaseModel) -> CrawlerRunConfig:
    instruction = T(f".prompts:kaggle_crawler.{instruction_name}").r()
    return CrawlerRunConfig(
        word_count_threshold=200,
        delay_before_return_html=3.0,
        page_timeout=30000,
        wait_until="networkidle",
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider=Provider, api_token=api_token, base_url=base_url),
            schema=schema.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            apply_chunk=False,
        ),
        cache_mode=CacheMode.BYPASS,
    )


async def crawl_page_with_retry(
    crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig, page_type: str, competition_name: str
) -> Optional[Dict[str, Any]]:
    """带重试机制的页面爬取函数"""
    retry_count = 0
    last_error = None

    while retry_count < MAX_RETRIES:
        try:
            print(f"正在爬取 {competition_name} 的 {page_type} 页面... (尝试 {retry_count + 1}/{MAX_RETRIES})")
            result = await crawler.arun(url=url, config=config)

            if result.extracted_content and result.extracted_content != "[]":
                extracted_data = (
                    json.loads(result.extracted_content)
                    if isinstance(result.extracted_content, str)
                    else result.extracted_content
                )
                if extracted_data and len(extracted_data) > 0:
                    print(f"✅ 成功爬取 {competition_name} 的 {page_type} 页面")
                    return extracted_data[0]

            # 如果没有提取到内容，也算作失败
            raise ValueError(f"未能从 {page_type} 页面提取到有效内容")

        except Exception as e:
            last_error = e
            retry_count += 1
            print(f"❌ 爬取 {competition_name} 的 {page_type} 页面失败 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}")

            if retry_count < MAX_RETRIES:
                delay = RETRY_DELAY * (RETRY_BACKOFF ** (retry_count - 1))
                print(f"⏳ 等待 {delay} 秒后重试...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ 达到最大重试次数，放弃爬取 {competition_name} 的 {page_type} 页面")

    return None


async def crawl_discussion(
    crawler: AsyncWebCrawler,
    competition_name: str,
    discussion_links: List[str],
    save_path: str | Path,
    force: bool = False,
) -> None:
    for link in tqdm(discussion_links, desc=f"Crawling Discussions for {competition_name}"):
        discussion_idx = link.split("/")[-1]
        discussion_config = create_crawler_config("discussion", DiscussionPageInfo)
        result = await crawler.arun(url=link, config=discussion_config)

        if result.extracted_content and result.extracted_content != "[]":
            extracted_contents = json.loads(result.extracted_content)
            for j, extracted_content in enumerate(extracted_contents):
                title = extracted_content.get("title", "")
                content = extracted_content.get("content", "")
                comments = extracted_content.get("comments", [])
                if title == "" or content == "" or comments == []:
                    continue
                notebook_link = extracted_content.get("notebook_link", "")

                if isinstance(save_path, str):
                    save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                discussion_path = save_path / "discussion" / f"discussion_{discussion_idx}_{j}.md"
                notebook_path = save_path / "notebook" / f"notebook_{discussion_idx}_{j}"
                discussion_path.parent.mkdir(parents=True, exist_ok=True)
                notebook_path.parent.mkdir(parents=True, exist_ok=True)

                # download notebook
                if notebook_link and re.match(r"^https://www\.kaggle\.com/code/[^/]+/[^/]+$", notebook_link):
                    download_kaggle_notebook(notebook_link, notebook_path)

                with open(discussion_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n")
                    f.write(f"{content}\n\n")
                    if comments != []:
                        f.write("# Comments\n")
                        for idx, comment in enumerate(comments):
                            author = comment.get("author", "Unknown")
                            comment_content = comment.get("content", "")
                            f.write(f"**{idx}. {author}:**\n{comment_content}\n\n")


async def main():
    browser_config = BrowserConfig(
        verbose=False, headless=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    with open("scripts/kaggle_competitions.yml", "r") as file:
        data = yaml.safe_load(file)["competitions"][:100]

    for i, item in tqdm(enumerate(data), desc="Processing Competitions"):
        name = item["link"].split("/")[-1]
        if item["solutions"] is None:
            continue
        solutions = item["solutions"][:3]
        # discussion: https://www.kaggle.com/c/{competition_name}/discussion/{number}
        # notebook: https://www.kaggle.com/code/{username}/{notebook_name}

        discussion_links = [
            sol["link"] for sol in solutions if sol["link"].startswith(f"https://www.kaggle.com/c/{name}/discussion/")
        ]

        save_path = Path(f"/data/userdata/v-xuminrui/knowledge/knowledge_v1/{name}")
        if save_path.exists():
            continue

        async with AsyncWebCrawler(config=browser_config) as crawler:
            await crawl_discussion(
                crawler=crawler,
                competition_name=name,
                discussion_links=discussion_links,
                save_path=save_path,
            )


if __name__ == "__main__":
    asyncio.run(main())
