"""
Dataset SFT Conversion Pipeline (Phase 2: SFT Conversion).
Workflow: Load migrated dataset → Schema Analysis → Intelligent Routing → Convert to Alpaca format

Prerequisites: Run test_pipeline.py first to migrate dataset to ./datasets/raw/
"""

import json
import os
from pathlib import Path

from rdagent.components.data import convert_to_sft

# Configuration
DATASETS_ROOT = Path("./datasets/raw")
OUTPUT_DIR = Path("./datasets/sft")
TASK_DESCRIPTION = "数学推理数据集"  # 需要与 test_pipeline.py 保持一致

print("=" * 70)
print("SFT 转换流程 (Phase 2: 数据转换与清洗)")
print("=" * 70)
print(f"数据集根目录: {DATASETS_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"任务描述: {TASK_DESCRIPTION}")
print("=" * 70)


def find_latest_dataset(datasets_root: Path) -> Path:
    """查找最新迁移的数据集目录"""
    if not datasets_root.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {datasets_root}")

    # 获取所有子目录
    subdirs = [d for d in datasets_root.iterdir() if d.is_dir()]

    if not subdirs:
        raise FileNotFoundError(f"未找到任何数据集: {datasets_root}")

    # 按修改时间排序，返回最新的
    latest_dataset = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_dataset


def test_sft_conversion():
    """测试 SFT 转换流程（智能分流）"""

    # Step 1: 查找最新数据集
    print("\n[Step 1/3] 查找迁移后的数据集...")

    try:
        dataset_path = find_latest_dataset(DATASETS_ROOT)
        print(f"✅ 找到数据集: {dataset_path.name}")
        print(f"   路径: {dataset_path}")
        print(f"   修改时间: {dataset_path.stat().st_mtime}")
    except Exception as e:
        print(f"❌ 未找到数据集: {e}")
        print(f"\n提示: 请先运行 test_pipeline.py 下载并迁移数据集")
        return False

    # Step 2: 准备输出路径
    print("\n[Step 2/3] 准备输出路径...")

    output_file = OUTPUT_DIR / f"{dataset_path.name}_alpaca.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"✅ 输出路径准备完成")
    print(f"   输出文件: {output_file}")

    # Clean checkpoint before conversion
    checkpoint_file = Path("sft_checkpoint.json")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"   🧹 清理 checkpoint: {checkpoint_file}")

    # Step 3: SFT 转换（智能分流）
    print("\n[Step 3/3] SFT 转换（智能分流系统）...")
    print("=" * 70)
    print("智能分流说明:")
    print("  - 轻量路径 (Light Path): 标准 Q&A 数据 → 简单转换 + 去重 + 并行质量评分")
    print("  - 重度路径 (Heavy Path): 混乱数据 → 去重 + 直接并行 LLM 转换")
    print("  - 系统自动根据数据质量选择路径")
    print("=" * 70)

    try:
        result = convert_to_sft(
            input_path=str(dataset_path),
            output_file=str(output_file),
            task_description=TASK_DESCRIPTION,
        )

        # 验证结果
        print("\n✅ 转换完成!")
        print("=" * 70)
        print("转换统计:")
        print(f"  处理路径: {result.get('processing_path', 'unknown').upper()}")
        print(f"  成功状态: {result['success']}")
        print(f"  输入样本: {result['stats'].get('total_rows', 0)}")
        print(f"  输出样本: {result['stats'].get('successful_rows', 0)}")
        print(f"  质量分数: {result['stats'].get('quality_score', 0):.2f}")
        print("=" * 70)

        # 检查输出文件
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)

            print(f"\n📄 输出文件验证:")
            print(f"  文件路径: {output_file}")
            print(f"  样本总数: {len(output_data)}")
            print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f}MB")
            print(f"  格式验证: {'✓' if all('instruction' in s and 'output' in s for s in output_data) else '✗'}")

            # 显示示例
            if output_data:
                print(f"\n📝 示例样本 (前 3 个):")
                for i, sample in enumerate(output_data[:3]):
                    print(f"\n  样本 {i+1}:")
                    print(f"    instruction: {sample['instruction'][:80]}...")
                    if sample.get("input"):
                        print(f"    input: {sample['input'][:60]}...")
                    print(f"    output: {sample['output'][:80]}...")
                    if "metadata" in sample:
                        print(f"    metadata: {sample['metadata']}")

            # 数据质量统计
            if output_data:
                avg_instruction_len = sum(len(s["instruction"]) for s in output_data) / len(output_data)
                avg_output_len = sum(len(s["output"]) for s in output_data) / len(output_data)
                has_metadata = sum(1 for s in output_data if "metadata" in s)

                print(f"\n📊 数据质量统计:")
                print(f"  平均 instruction 长度: {avg_instruction_len:.0f} 字符")
                print(f"  平均 output 长度: {avg_output_len:.0f} 字符")
                print(f"  包含 metadata: {has_metadata}/{len(output_data)} 样本")

        return result["success"]

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """运行完整的 SFT 转换流程"""

    success = test_sft_conversion()

    # 总结
    print("\n" + "=" * 70)
    if success:
        print("✅ SFT 转换流程完成!")
        print("=" * 70)
        print("下一步建议:")
        print("  1. 检查输出文件质量")
        print("  2. 使用输出文件进行 LoRA/SFT 训练")
        print(f"  3. 输出文件位置: {OUTPUT_DIR}")
    else:
        print("❌ SFT 转换失败!")
        print("=" * 70)
        print("请检查:")
        print("  1. 是否已运行 test_pipeline.py 迁移数据集?")
        print("  2. 数据集格式是否正确?")
        print("  3. LLM API 是否配置正确?")
    print("=" * 70)


if __name__ == "__main__":
    main()
