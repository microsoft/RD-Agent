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
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field


class KaggleCompetitionInfo(BaseModel):
    overview: str = Field(..., description="Overview/description of the competition.")
    evaluation: str = Field(..., description="Evaluation method and metrics for the competition.")
    submission_file: str = Field(..., description="Submission file and format for the competition.")
    other_important_info: str = Field(..., description="Other important information for the competition.")
    tags: list[str] = Field(
        ..., description="Tags of the competition，should give the full tag list, not just the first few tags."
    )


Provider = "gpt-4.1"
api_token = "sk-1234"
base_url = "http://10.150.240.117:38888"


class DataPageInfo(BaseModel):
    dataset_description: str = Field(
        ...,
        description="Extract all detailed descriptions about the dataset from the Kaggle competition data page, including dataset structure, file lists, field definitions, data types, data sources, and any usage notes. The content should be presented in complete and structured Markdown format, including all original formatting such as headers, tables, lists, etc. You need to cover as much content as possible from the page and not miss any information. Except for ## Dataset Description, the maximum heading level for other sections should be ###. In English.",
    )


MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
RETRY_BACKOFF = 2  # 指数退避倍数


def create_overview_config():
    return CrawlerRunConfig(
        word_count_threshold=1,
        delay_before_return_html=3,
        page_timeout=30000,  # 增加页面超时时间
        wait_until="networkidle",  # 等待网络空闲
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider=Provider, api_token=api_token, base_url=base_url),
            schema=KaggleCompetitionInfo.model_json_schema(),
            extraction_type="block",
            instruction="""从爬取的内容中提取Kaggle竞赛的以下信息：
            1. Overview (概览): 竞赛的简介和目标
            2. Evaluation (评估方法): 竞赛提交的评估标准和指标
            3. Submission File (提交文件): 竞赛提交的文件格式和要求
            4. Other Important Info (其他重要信息): 竞赛的其他重要信息
            5. Tags (标签): 竞赛的标签,以列表形式返回,应该返回所有的tags
            
            注意，以下内容不需要提取：
            1. Timeline (时间线): 竞赛的重要日期，包括开始时间、结束时间等
            2. Citation (引用): 如何引用该竞赛或数据集的信息
            3. Prizes (奖品): 竞赛奖品信息 
            4. Competition Host
            5. Participation

            
            请仔细查找页面中所有相关部分，不要遗漏任何信息。提取的内容应该尽量保持不更改原始页面的任何内容，以markdown格式返回,返回的最大格式为##，注意分隔符（比如换行符等）。
            
            格式要求：
            - 返回的最大标题格式为##（二级标题）
            - 保留原文中所有HTML标题符号，转换成对应的markdown标题格式（h1->##, h2->##, h3->###, h4->####等，但最高不超过##）
            - 保持原始内容的分隔符（换行符、空行等）
            - 保留所有列表、代码块、表格等格式
            - 不要修改或改写原始内容，只进行格式转换
            - 不要保留任何HTML标签，以及外部链接
            """,
        ),
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["Timeline", "Citation", "Prizes"],
    )


def create_data_config():
    return CrawlerRunConfig(
        word_count_threshold=1,
        delay_before_return_html=3.0,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider=Provider, api_token=api_token, base_url=base_url),
            schema=DataPageInfo.model_json_schema(),
            extraction_type="schema",
            instruction="""从data页面中提取所有与数据集相关的所有信息，你需要尽可能多的覆盖页面中的所有内容，不要遗漏任何信息。
            
            **提取原则**：
            1. 提取所有与数据集相关的核心信息
            2. 包括但不限于：数据集描述、文件说明、格式信息、字段定义等
            3. 自动识别页面中存在的数据相关章节,比如files,columns等，如果页面中没有某个章节，就不要包含
            
            **排除以下内容**：
            - License/许可证信息
            - Prizes/奖品信息
            - Citation/引用信息
            - License/许可证信息
            
            **输出要求**：
            - 严格按照markdown格式输出
            - 使用##、###等标题层级
            - 保留所有格式：列表、表格、代码等
            - 每个章节都应该是完整的markdown文本
            - 不要保留任何HTML标签，以及外部链接
            
           
            ```""",
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


def create_competition_folder(competition_name: str, save_path: Path) -> Path:
    """为比赛创建文件夹"""
    folder_path = save_path / competition_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def save_competition_files(competition_name: str, final_result: Dict[str, Any], save_path: Path) -> None:
    """保存比赛的JSON和Markdown文件到指定文件夹"""
    folder_path = create_competition_folder(competition_name, save_path)

    # Create final JSON output
    final_json = {
        competition_name: {
            "overview": final_result.get("overview", ""),
            "evaluation": final_result.get("evaluation", ""),
            "tags": final_result.get("tags", ""),
            "submission_file": final_result.get("submission_file", ""),
            "other_important_info": final_result.get("other_important_info", ""),
            "dataset_description": final_result.get("dataset_description", ""),
        }
    }

    # Save JSON file
    json_filename = folder_path / f"{competition_name}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    # Create markdown content
    markdown_content = f"# {competition_name.replace('-', ' ').title()}\n\n"

    for key in ["overview", "tags", "evaluation", "submission_file", "other_important_info", "dataset_description"]:
        result = final_result.get(key)
        if key == "tags":
            markdown_content += f"\n## {key.capitalize()}\n{result}\n\n"
        elif result and key != "dataset_description":
            markdown_content += f"\n{result}\n\n"
        elif result and key == "dataset_description":
            if result.startswith("###"):
                result = result.replace("###", "##", 1)
            markdown_content += f"\n{result}\n\n"

    # Save markdown file
    markdown_filename = folder_path / f"{competition_name}.md"
    with open(markdown_filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ 已保存 {competition_name} 的文件到文件夹: {folder_path}")


def write_failed_competition_to_csv(competition_name: str, error_message: str, save_path: Path) -> None:
    """将失败的比赛信息写入CSV文件"""
    csv_file = save_path / "failed_competitions.csv"

    # 检查CSV文件是否存在，如果不存在则创建并写入表头
    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(["Competition Name", "Failed Time", "Error Message"])

        # 写入失败记录
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([competition_name, current_time, error_message])

    print(f"❌ 已记录失败比赛到CSV: {competition_name}")


async def process_single_competition(crawler: AsyncWebCrawler, competition_name: str, save_path: Path) -> bool:
    """处理单个比赛的爬取"""
    print(f"\n🚀 开始处理比赛: {competition_name}")

    base_url = f"https://www.kaggle.com/competitions/{competition_name}"

    pages_config = [
        ("overview", f"{base_url}/overview", create_overview_config()),
        ("data", f"{base_url}/data", create_data_config()),
    ]

    final_result = {}
    success_count = 0
    error_messages = []

    for page_type, url, page_config in pages_config:
        data = await crawl_page_with_retry(crawler, url, page_config, page_type, competition_name)

        if data:
            success_count += 1
            if page_type == "overview":
                final_result["overview"] = data.get("overview", "")
                final_result["evaluation"] = data.get("evaluation", "")
                final_result["submission_file"] = data.get("submission_file", "")
                final_result["other_important_info"] = data.get("other_important_info", "")
                final_result["tags"] = data.get("tags", "")
            elif page_type == "data":
                final_result["dataset_description"] = data.get("dataset_description", "")
        else:
            error_messages.append(f"Failed to crawl {page_type} page")

    # 只有当所有页面都成功爬取时才保存文件，否则记录到失败CSV
    if success_count == len(pages_config):
        save_competition_files(competition_name, final_result, save_path)
        print(f"✅ 比赛 {competition_name} 处理完成 (成功: {success_count}/{len(pages_config)} 页面)")
        return True
    else:
        error_msg = f"部分页面爬取失败 (成功: {success_count}/{len(pages_config)}). 错误: {'; '.join(error_messages)}"
        write_failed_competition_to_csv(competition_name, error_msg, save_path)
        print(f"❌ 比赛 {competition_name} 处理失败 - {error_msg}")
        return False


async def process_competition_list(competition_list: List[str], save_path: Optional[str] = None) -> Dict[str, bool]:
    """批量处理比赛列表"""
    # 设置保存路径
    if save_path is None:
        save_path_obj = Path.cwd()
    else:
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)

    print(f"📁 文件保存路径: {save_path_obj.absolute()}")

    browser_config = BrowserConfig(
        verbose=False, headless=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    results = {}

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print(f"📊 开始批量处理 {len(competition_list)} 个比赛")

        for i, competition_name in enumerate(competition_list, 1):
            print(f"\n{'='*50}")
            print(f"处理进度: {i}/{len(competition_list)}")

            success = await process_single_competition(crawler, competition_name, save_path_obj)
            results[competition_name] = success

            # 在比赛之间添加短暂延迟，避免请求过于频繁
            if i < len(competition_list):
                await asyncio.sleep(2)

    return results


async def main():
    """主函数 - 可以处理单个比赛或比赛列表"""

    # 示例：处理单个比赛
    single_competition = "us-patent-phrase-to-phrase-matching"

    # 示例：处理比赛列表（你可以根据需要修改这个列表）
    df = pd.read_csv(
        "/data/userdata/v-zhangyifei/MSRA/github/knowledge/github_competitions/比赛小节/competitions_extracted_info.csv"
    )
    competition_list = df["link_slug"].tolist()
    print(f"处理{len(competition_list)}个比赛")

    # 设置自定义保存路径（可以修改为你想要的路径）
    custom_save_path = (
        "/data/userdata/v-zhangyifei/MSRA/github/knowledge/kaggle_competitions"  # 设置为None则使用当前目录
    )

    # 选择处理模式
    print("选择处理模式:")
    print("1. 处理单个比赛")
    print("2. 批量处理比赛列表")

    # 这里你可以修改为直接指定模式，或者从命令行参数获取
    mode = int(input("请输入处理模式: "))

    if mode == 1:
        # 处理单个比赛
        browser_config = BrowserConfig(
            verbose=False, headless=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        # 设置保存路径
        if custom_save_path is None:
            save_path_obj = Path.cwd()
        else:
            save_path_obj = Path(custom_save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)

        print(f"📁 文件保存路径: {save_path_obj.absolute()}")

        async with AsyncWebCrawler(config=browser_config) as crawler:
            success = await process_single_competition(crawler, single_competition, save_path_obj)
            if success:
                print(f"\n🎉 比赛 {single_competition} 处理成功!")
            else:
                print(f"\n😞 比赛 {single_competition} 处理失败!")

    else:
        # 批量处理比赛列表
        results = await process_competition_list(competition_list, custom_save_path)

        # 输出处理结果统计
        print(f"\n{'='*50}")
        print("📋 处理结果统计:")
        successful = sum(results.values())
        total = len(results)

        print(f"✅ 成功: {successful}/{total}")
        print(f"❌ 失败: {total - successful}/{total}")

        if successful < total:
            print("\n失败的比赛:")
            for comp, success in results.items():
                if not success:
                    print(f"  - {comp}")
            print(
                f"\n📄 失败记录已保存到: {Path(custom_save_path if custom_save_path else '.') / 'failed_competitions.csv'}"
            )


def process_competitions(competition_list: List[str], save_path: Optional[str] = None) -> None:
    """提供给外部调用的函数"""
    asyncio.run(process_competition_list(competition_list, save_path))


if __name__ == "__main__":
    asyncio.run(main())
