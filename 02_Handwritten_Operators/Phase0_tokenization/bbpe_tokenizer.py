# 字节级 (Byte-level) BPE 分词器 — 现代工业标准
#
# 从 UTF-8 字节流开始合并，而非字符。
# 核心技巧：bytes_to_unicode() 将 256 个字节值映射为可见 Unicode 字符，
# 使正则和 BPE 合并逻辑无需感知原始字节的语义。
#
# 与字符级 BPE 的核心区别：
#   - 基础单元是字节（0–255），初始词表固定 256 个
#   - 永远不会有 OOV：任何 Unicode 字符都是 1–4 字节的组合
#   - 无损编解码：空格、标点、格式完全保留
#
# 使用方式：
#   python bbpe_tokenizer.py          # 运行冒烟测试
#   from bbpe_tokenizer import BBPETokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import List

from base_bpe import BaseBPETokenizer
from byte_to_unicode_map import bytes_to_unicode


class BBPETokenizer(BaseBPETokenizer):
    """字节级 BPE 分词器。

    继承 BaseBPETokenizer，在 train/encode/decode 入口叠加
    UTF-8 字节 ↔ 可见字符的转换步骤。
    """

    def __init__(self):
        super().__init__()
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    # ── 钩子实现 ────────────────────────────────────────────────────

    def _preprocess_train(self, text: str) -> str:
        """训练前：UTF-8 编码 → 字节流 → 可见字符映射。"""
        raw_bytes = text.encode("utf-8")
        return "".join(self.byte_encoder[b] for b in raw_bytes)

    def _preprocess_encode(self, text: str) -> str:
        """编码前：同上。"""
        return self._preprocess_train(text)

    def _postprocess_decode(self, text: str) -> str:
        """解码后：可见字符 → 字节流 → UTF-8 解码。"""
        raw_bytes = bytes([self.byte_decoder[ch] for ch in text])
        return raw_bytes.decode("utf-8", errors="replace")

    def _init_vocab(self, chars: List[str]) -> int:
        """字节级 BPE：从 byte_encoder 的 256 个可见字符初始化词表。

        忽略 chars 参数（即使训练数据中某些字节未出现，也需要全量 256 个
        以保证永远不 OOV）。
        """
        initial_chars = sorted(set(self.byte_encoder.values()))
        for i, char in enumerate(initial_chars):
            self.vocab[i] = char
            self.inverse_vocab[char] = i
        return len(initial_chars)

    def _on_unknown_symbol(self, sym: str) -> int:
        """BBPE 理论上不会遇到未知符号（256 字节全覆盖）。

        如果触发，说明 byte_encoder 映射表出了问题——抛出异常而非静默丢弃。
        """
        raise KeyError(
            f"BBPE 遇到词表外的符号 {repr(sym)}——"
            "这通常意味着 byte_encoder 映射不完整。"
        )


from typing import List  # noqa: E402 — 供 __main__ 签名提示

if __name__ == "__main__":
    train_text = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3

    tokenizer = BBPETokenizer()
    tokenizer.train(train_text, num_merges=10)
    print(f"[*] 词表内容（前 20 项）: ")
    for i, (k, v) in enumerate(tokenizer.vocab.items()):
        if i >= 20:
            break
        print(f"    {k}: {repr(v)}")

    # 测试：含特殊 token、英文、中文、Emoji
    test_text = "<think> I'm lower than the newestest! </think> \n 你好!我叫孟志泉 🚀"

    print(f"\n[原始文本] {test_text}")

    ids = tokenizer.encode(test_text)
    print(f"[编码 ID] ({len(ids)} tokens): {ids}")

    decoded = tokenizer.decode(ids)
    print(f"[解码文本] {decoded}")

    assert test_text == decoded, "测试失败：编解码不一致！"
    print("\n[*] 验证通过：编解码完全一致！")
