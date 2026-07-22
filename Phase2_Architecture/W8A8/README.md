# W8A8 量化：对称/非对称量化 + SmoothQuant — 手撕实现

本目录从零实现 LLM INT8 量化的完整链路：量化原语（对称/非对称）→ 离群值灾难 → SmoothQuant 难度迁移，用三阶段逐层揭示大模型 W8A8 量化的核心原理。

## 目录结构

```
W8A8/
├── README.md                      # 本文件 — 目录索引与学习路径
├── quant_primitives.py            #   量化原语：TensorQuantizer（对称/非对称、quantize/dequantize）
├── w8a8_gemm_mock.py              #   离群值灾难演示：W8A16 vs Naive W8A8 MSE 对比
├── smooth_quant.py                #   SmoothQuant 实现：平滑因子计算 + 难度迁移 + 全方法对比
├── SmoothQuant_Paper_Notes.md     #   论文精读：算法原理 → 实现细节 → 实验 → 面试 Q&A
└── run_all_tests.py               #   集成测试入口：Day 36 → Day 37 → Day 38 顺序运行
```

## 组件速览

| 文件 | 一句话定位 | 读者 |
| :--- | :--- | :--- |
| [quant_primitives.py](quant_primitives.py) | `TensorQuantizer` 类：对称/非对称量化的 scale/zp 计算 + quantize/dequantize | 先看这里理解量化基本操作 |
| [w8a8_gemm_mock.py](w8a8_gemm_mock.py) | 注入 outlier 后 W8A16 vs Naive W8A8 的 MSE 对比——为什么激活值不能直接量化 | 理解离群值如何摧毁精度 |
| [smooth_quant.py](smooth_quant.py) | SmoothQuant 完整实现：平滑因子计算 → X/W 难度迁移 → W8A8 量化 → 全方法对比 | 理解 SmoothQuant 怎么救回来 |
| [SmoothQuant_Paper_Notes.md](SmoothQuant_Paper_Notes.md) | 论文精读：channel-wise outlier → 等价变换 → α 参数 → O1/O2/O3 → 面试高频问答 | 追论文细节和面试准备 |
| [run_all_tests.py](run_all_tests.py) | 一键运行 Day 36→37→38 全部测试，输出量化方法综合对比表 | 验证整体链路 |

## 快速运行

```bash
# 单个测试
python quant_primitives.py
python w8a8_gemm_mock.py
python smooth_quant.py

# 完整链路
python run_all_tests.py
```

## 建议学习路径

| 顺序 | 文件 | 核心问题 |
| :--- | :--- | :--- |
| 1 | [quant_primitives.py](quant_primitives.py) | 对称 vs 非对称量化的 scale/zero-point 怎么算？为什么对称量化 zp=0？ |
| 2 | [w8a8_gemm_mock.py](w8a8_gemm_mock.py) | 注入一个 150× 的 outlier 后，Naive W8A8 的 MSE 为什么爆炸 2000 倍？ |
| 3 | [smooth_quant.py](smooth_quant.py) | SmoothQuant 怎么通过 `X/S` 和 `W*S` 把难度从 activation 迁移到 weight？ |
| 4 | [SmoothQuant_Paper_Notes.md](SmoothQuant_Paper_Notes.md) | 为什么 activation per-channel quantization 精度好但硬件不友好？α 参数怎么选？ |

## 文档约定

| 约定 | 说明 |
| :--- | :--- |
| **`README.md`** | 精简索引：目录结构 + 组件速览 + 学习路径 |
| **代码注释** | 中文注释 + 公式标注（如 `// x_q = clamp(round(x/S + ZP), -127, 127)`） |
| **冒烟测试** | 每个 `.py` 自带 `__main__` 块，含 MSE 断言验证 |
| **论文笔记** | `SmoothQuant_Paper_Notes.md` 按论文结构组织，含面试 Q&A |

## 核心链路速览

```
FP32 张量
  │
  ├─ Day 36: quant_primitives.py ──→ INT8 量化/反量化（MSE ≈ 9e-05，几乎无损）
  │
  ├─ Day 37: w8a8_gemm_mock.py ──→ 注入 outlier → Naive W8A8 MSE 爆炸（3484 vs 1.6）
  │                                   原因：outlier 拉大 scale，正常值被抹平成 0
  │
  └─ Day 38: smooth_quant.py ────→ SmoothQuant 平滑 X/W → W8A8 MSE 降至 31.76
                                    原理：Y = X·W = (X/S)·(S·W)，难度从 A 迁移到 W
```
