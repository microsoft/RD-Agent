"""
Dataset preparation script (Phase 1: Data Collection).
Workflow: Search → Download → Inspect → LLM Filter → Migrate to ./datasets/raw/

For SFT conversion, use sft_converter.py (Phase 2).
"""

import shutil
from pathlib import Path

from rdagent.components.data import (
    DatasetInspector,
    DatasetManager,
    DatasetSearchAgent,
)

# Configuration
TASK_DESCRIPTION = "数学推理数据集"
TEMP_STAGING_DIR = "/tmp/dataset_staging"
FINAL_STORAGE_DIR = "./datasets"
MAX_CANDIDATES = 5

print("=" * 70)
print("数据集准备流程 (Phase 1: 数据采集与迁移)")
print("=" * 70)
print(f"任务: {TASK_DESCRIPTION}")
print(f"临时目录: {TEMP_STAGING_DIR}")
print(f"最终存储: {FINAL_STORAGE_DIR}/raw/")
print("=" * 70)

# Step 1: Search and Download to temporary directory
print("\n[Step 1/4] 搜索并下载数据集...")
agent = DatasetSearchAgent()

try:
    search_result = agent.search_and_download(
        task_description=TASK_DESCRIPTION,
        out_dir=TEMP_STAGING_DIR,
        max_candidates=MAX_CANDIDATES,
    )

    print(f"✅ 下载完成!")
    print(f"  数据集ID: {search_result['dataset_id']}")
    print(f"  临时路径: {search_result['dataset_path']}")
    print(f"  置信度: {search_result['confidence']:.2f}")
    print(f"  理由: {search_result['reason']}")

except Exception as e:
    print(f"❌ 搜索下载失败: {e}")
    exit(1)

# Step 2: Inspect dataset structure
print("\n[Step 2/4] 检查数据集结构...")
inspector = DatasetInspector()

try:
    inspect_result = inspector.inspect(search_result["dataset_path"], sample_size=10)

    print(f"✅ 检查完成!")
    print(f"  可加载: {inspect_result['loadable']}")
    print(f"  样本数: {inspect_result['total_samples']}")
    print(f"  列名: {inspect_result['columns']}")

    if inspect_result["issues"]:
        print(f"  ⚠️  发现问题:")
        for issue in inspect_result["issues"]:
            print(f"    - {issue}")

except Exception as e:
    print(f"❌ 数据集检查失败: {e}")
    exit(1)

# Step 3: LLM analyzes which files are useful for SFT
print("\n[Step 3/4] LLM分析文件（识别垃圾文件）...")

try:
    file_analysis = inspector.analyze_files_for_sft(
        dataset_path=search_result["dataset_path"],
        task_description=TASK_DESCRIPTION,
    )

    print(f"✅ 文件分析完成!")
    print(f"  有用文件: {len(file_analysis['useful_files'])}")
    print(f"  垃圾文件: {len(file_analysis['junk_files'])}")
    print(f"  原始大小: {file_analysis['total_size_mb']:.2f}MB")
    print(f"  清理后: {file_analysis['size_after_cleanup_mb']:.2f}MB")
    print(f"  节省空间: {file_analysis['space_saved_mb']:.2f}MB")

    if file_analysis["junk_files"]:
        print(f"\n  🗑️  将删除的垃圾文件:")
        for junk_file in file_analysis["junk_files"][:10]:  # 只显示前10个
            print(f"    - {junk_file}")
        if len(file_analysis["junk_files"]) > 10:
            print(f"    ... 还有 {len(file_analysis['junk_files']) - 10} 个文件")

except Exception as e:
    print(f"❌ 文件分析失败: {e}")
    print(f"  将使用规则过滤作为后备方案")
    # 如果LLM失败，analyze_files_for_sft会自动fallback到规则过滤
    import traceback

    traceback.print_exc()

# Step 4: Selective migration to ./datasets/raw/
print("\n[Step 4/4] 迁移到永久存储（只复制有用文件）...")
manager = DatasetManager(permanent_root=FINAL_STORAGE_DIR)

try:
    migration_result = manager.migrate_dataset_selective(
        source_path=search_result["dataset_path"],
        dataset_id=search_result["dataset_id"],
        file_analysis=file_analysis,
    )

    print(f"✅ 迁移完成!")
    print(f"  目标路径: {migration_result['target_path']}")
    print(f"  复制文件数: {len(migration_result['copied_files'])}")
    print(f"  跳过文件数: {len(migration_result['skipped_files'])}")
    print(f"  节省空间: {migration_result['space_saved_mb']:.2f}MB")

except Exception as e:
    print(f"❌ 迁移失败: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ 数据集准备完成！已迁移至永久存储")
