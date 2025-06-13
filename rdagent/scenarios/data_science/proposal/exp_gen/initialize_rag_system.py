
"""
RAG系统初始化脚本
用于首次部署时初始化RAG系统，加载idea_v4.json并构建索引
"""

import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.proposal.exp_gen.rag_hybrid_2 import HybridRAGSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_idea_data(ideas: Dict[str, Any]) -> tuple[bool, list]:
    """
    验证idea数据的格式和内容
    
    Returns:
        (is_valid, errors): 是否有效，错误列表
    """
    errors = []
    valid_labels = {'SCENARIO_PROBLEM', 'FEEDBACK_PROBLEM'}
    required_fields = ['problem']  # 至少需要problem字段
    
    for idea_name, idea_data in ideas.items():
        if not isinstance(idea_data, dict):
            errors.append(f"Idea '{idea_name}' 不是字典格式")
            continue
        
        # 检查必需字段
        for field in required_fields:
            if field not in idea_data or not idea_data[field]:
                errors.append(f"Idea '{idea_name}' 缺少必需字段 '{field}'")
        
        # 检查label有效性
        if 'label' in idea_data:
            label = idea_data['label'].upper()
            if label and label not in valid_labels:
                errors.append(f"Idea '{idea_name}' 的label '{label}' 无效，应为 {valid_labels}")
    
    return len(errors) == 0, errors


def analyze_idea_pool(ideas: Dict[str, Any]) -> Dict[str, Any]:
    """分析idea池的统计信息"""
    stats = {
        'total_count': len(ideas),
        'label_distribution': {},
        'field_coverage': {},
        'avg_text_length': 0,
        'missing_fields': {}
    }
    
    # 统计label分布
    label_counts = {'SCENARIO_PROBLEM': 0, 'FEEDBACK_PROBLEM': 0, 'UNKNOWN': 0}
    
    # 统计字段覆盖率
    field_counts = {}
    total_text_length = 0
    
    for idea_name, idea_data in ideas.items():
        if isinstance(idea_data, dict):
            # 统计label
            label = idea_data.get('label', '').upper()
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts['UNKNOWN'] += 1
            
            # 统计字段
            for field in ['problem', 'method', 'context', 'reason', 'label']:
                if field not in field_counts:
                    field_counts[field] = 0
                if field in idea_data and idea_data[field]:
                    field_counts[field] += 1
            
            # 计算文本长度
            text_length = sum(len(str(v)) for v in idea_data.values() if v)
            total_text_length += text_length
    
    stats['label_distribution'] = label_counts
    stats['field_coverage'] = {
        field: f"{(count/len(ideas)*100):.1f}%" 
        for field, count in field_counts.items()
    }
    stats['avg_text_length'] = int(total_text_length / len(ideas)) if ideas else 0
    
    return stats


def initialize_rag_system(
    idea_pool_path: str = "./idea_v4.json",
    cache_dir: str = "./hypothesis_rag_cache",
    force_rebuild: bool = False
):
    """
    初始化RAG系统
    
    Args:
        idea_pool_path: idea池JSON文件路径
        cache_dir: 缓存目录
        force_rebuild: 是否强制重建索引
    """
    
    # 检查idea文件是否存在
    if not os.path.exists(idea_pool_path):
        logger.error(f"找不到idea文件: {idea_pool_path}")
        return False
    
    logger.info(f"加载idea池文件: {idea_pool_path}")
    
    # 加载并验证数据
    try:
        with open(idea_pool_path, 'r', encoding='utf-8') as f:
            ideas = json.load(f)
        logger.info(f"成功加载 {len(ideas)} 个ideas")
    except Exception as e:
        logger.error(f"加载idea文件失败: {e}")
        return False
    
    # 验证数据
    is_valid, errors = validate_idea_data(ideas)
    if not is_valid:
        logger.warning(f"发现 {len(errors)} 个数据问题:")
        for error in errors[:10]:  # 只显示前10个错误
            logger.warning(f"  - {error}")
        if len(errors) > 10:
            logger.warning(f"  ... 还有 {len(errors) - 10} 个错误")
    
    # 分析数据
    logger.info("分析idea池...")
    stats = analyze_idea_pool(ideas)
    logger.info(f"- 总数量: {stats['total_count']}")
    logger.info(f"- Label分布: {stats['label_distribution']}")
    logger.info(f"- 字段覆盖率: {stats['field_coverage']}")
    logger.info(f"- 平均文本长度: {stats['avg_text_length']} 字符")
    
    # 创建API后端
    try:
        api_backend = APIBackend()
        logger.info("API后端创建成功")
    except Exception as e:
        logger.error(f"创建API后端失败: {e}")
        logger.error("请检查API配置（如OPENAI_API_KEY等）")
        return False
    
    # 检查是否需要重建
    if force_rebuild and os.path.exists(cache_dir):
        logger.info("强制重建模式：清理现有缓存...")
        import shutil
        shutil.rmtree(cache_dir)
    
    # 创建RAG系统
    logger.info("初始化RAG系统...")
    try:
        rag_system = HybridRAGSystem(api_backend, cache_dir=cache_dir)
        
        # 获取当前状态
        current_stats = rag_system.get_statistics()
        logger.info(f"当前系统状态: {current_stats['total_ideas']} 个ideas")
        
        # 如果是空系统或强制重建，加载数据
        if current_stats['total_ideas'] == 0 or force_rebuild:
            logger.info("开始构建索引（这可能需要几分钟）...")
            start_time = datetime.now()
            
            # 批量添加ideas
            batch_size = 100
            for i in range(0, len(ideas), batch_size):
                batch_end = min(i + batch_size, len(ideas))
                batch = dict(list(ideas.items())[i:batch_end])
                
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(ideas) + batch_size - 1)//batch_size}")
                rag_system.add_ideas(batch, update_cache=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"索引构建完成，耗时: {duration:.2f} 秒")
        else:
            logger.info("系统已包含数据，跳过初始化")
        
        # 显示最终状态
        final_stats = rag_system.get_statistics()
        logger.info("初始化完成！系统状态:")
        logger.info(f"- Ideas总数: {final_stats['total_ideas']}")
        logger.info(f"- Embeddings总数: {final_stats['total_embeddings']}")
        logger.info(f"- BM25索引: {'已建立' if final_stats['index_status']['bm25'] else '未建立'}")
        logger.info(f"- FAISS索引: {'已建立' if final_stats['index_status']['faiss'] else '未建立'}")
        logger.info(f"- 缓存目录: {cache_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"初始化RAG系统失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_system(cache_dir: str = "./hypothesis_rag_cache"):
    """测试RAG系统是否正常工作"""
    logger.info("\n测试RAG系统...")
    
    try:
        
        api_backend = APIBackend()
        rag_system = HybridRAGSystem(api_backend, cache_dir=cache_dir)
        
        # 测试检索
        test_problems = {
    "Hidden class imbalance": {
        "problem": "The proportion of positive (cactus-present) images is considerably smaller than negatives, so a model trained with an unweighted loss will learn a decision surface that under-scores rare cactus patterns and inflates false-negative rates.",
        "reason": "ROC-AUC is sensitive to the ranking of minority positives; correcting the imbalance with class-balanced/focal loss or stratified sampling will make positive samples influence the gradient more strongly and lift AUC.",
        "label": "SCENARIO_PROBLEM"
    },
    "Unsupported 'region' argument in tifffile": {
        "problem": "The DataLoader uses TiffPage.asarray(region=...) to crop sub‑windows, but tifffile’s API does not accept a 'region' keyword, so every worker raises TypeError and training never starts.",
        "reason": "Fixing the call (e.g. load the full page or use tifffile.TiffFile.asarray(...) followed by numpy slicing) will let the model actually receive image data, enabling training/inference and thus any Dice optimisation at all.",
        "label": "FEEDBACK_PROBLEM"
    }
        }
        
        logger.info("测试检索功能...")
        results = rag_system.parallel_retrieve(test_problems, top_k=3)
        
        for problem_name, relevant_ideas in results.items():
            logger.info(f"\n问题: {problem_name}")
            logger.info(f"找到 {len(relevant_ideas)} 个相关ideas:")
            for i, idea in enumerate(relevant_ideas[:2], 1):
                logger.info(f"  {i}. {idea['idea_name']} (相似度: {idea['similarity_score']:.3f})")
                if idea.get('matching_aspects'):
                    logger.info(f"     匹配特征: {', '.join(idea['matching_aspects'])}")
                    logger.info(f"具体想法为：{idea}")        
        logger.info("\n测试成功！RAG系统工作正常。")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化RAG系统")
    parser.add_argument(
        "--idea-pool", 
        default="./idea_v4.json",
        help="idea池JSON文件路径 (默认: ./idea_v4.json)"
    )
    parser.add_argument(
        "--cache-dir",
        default="./hypothesis_rag_cache",
        help="缓存目录路径 (默认: ./hypothesis_rag_cache)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建所有索引"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="初始化后运行测试"
    )
    
    args = parser.parse_args()
    
    # 执行初始化
    success = initialize_rag_system(
        idea_pool_path=args.idea_pool,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild
    )
    
    if success and args.test:
        test_rag_system(args.cache_dir)
    
    sys.exit(0 if success else 1)