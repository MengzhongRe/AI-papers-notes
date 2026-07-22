# 字符级 (Character-level) BPE 分词器 — 适用于教学与面试
#
# 从训练数据中出现的字符开始合并，直到达到预设的 merge 轮数。
#
# 与字节级 BBPE 的核心区别：
#   - 基础单元是字符（Unicode code point），不是字节
#   - 遇到训练数据中未出现的字符时回退到 <UNK>
#   - 不需要 UTF-8 编码／解码步骤
#
# 使用方式：
#   python bpe_tokenizer.py          # 运行冒烟测试
#   from bpe_tokenizer import BPETokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from base_bpe import BaseBPETokenizer


class BPETokenizer(BaseBPETokenizer):
    """字符级 BPE 分词器。

    继承自 BaseBPETokenizer，不做任何字节映射，
    直接对字符序列进行 BPE 合并。
    """

    # 字符级不需要重写 _preprocess_train / _preprocess_encode / _postprocess_decode，
    # 基类默认行为就是直通（identity）。


if __name__ == "__main__":
    # 使用与 BBPE 相同的训练语料，便于对比两者的差异
    train_text = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, num_merges=10)

    # 字符级 BPE：只包含训练数据中的 11 个字符
    print(f"[*] 词表内容: {tokenizer.vocab}")

    # 测试 1：训练域内的文本 — 应能无损编解码
    test_in = "<think> low lower newest widest </think>"
    ids_in = tokenizer.encode(test_in)
    dec_in = tokenizer.decode(ids_in)
    print(f"\n[域内测试] {test_in}")
    print(f"  ID: {ids_in}")
    print(f"  => {dec_in}")
    assert test_in == dec_in, "域内编解码不一致！"

    # 测试 2：域外文本 — 未在训练数据中出现的字符回退为 <UNK>
    test_oov = "hello world"
    ids_oov = tokenizer.encode(test_oov)
    dec_oov = tokenizer.decode(ids_oov)
    print(f"\n[OOV 测试] {test_oov}")
    print(f"  ID: {ids_oov}")
    print(f"  => {dec_oov}")
    # 字符 'h', 'a' 不在训练数据中 → <UNK>
    assert "<UNK>" in dec_oov, "OOV 字符应回退为 <UNK>"

    print("\n[*] 字符级 BPE 验证通过！")
    print("[*] 提示: 对比 bbpe_tokenizer.py 观察 BBPE 如何免除 <UNK>。")
