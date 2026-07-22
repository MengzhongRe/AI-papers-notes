"""
Day 36-38 综合手撕代码测试
验证: 基础量化 -> 离群值灾难 -> SmoothQuant 救世主
"""

import sys
from quant_primitives import test_primitives
from w8a8_gemm_mock import test_outlier_collapse
from smooth_quant import compare_all_methods


def main():
    print("\n" + "=" * 80)
    print("LLM INT8 量化深度学习: Day 36-38 完整演义")
    print("=" * 80)

    # Day 36: 基础量化原语
    print("\n" + "▶" * 40)
    print("PART 1: Day 36 - 基础量化原语")
    print("▶" * 40)
    test_primitives()

    # Day 37: 离群值灾难
    print("\n" + "▶" * 40)
    print("PART 2: Day 37 - 离群值灾难(The Outlier Collapse)")
    print("▶" * 40)
    test_outlier_collapse()

    # Day 38: SmoothQuant 救世
    print("\n" + "▶" * 40)
    print("PART 3: Day 38 - SmoothQuant 数学救世")
    print("▶" * 40)
    compare_all_methods()

    print("\n" + "=" * 80)
    print("✨ 完成! 你已掌握大厂推理架构的核心量化技术 ✨")
    print("=" * 80)


if __name__ == "__main__":
    main()
