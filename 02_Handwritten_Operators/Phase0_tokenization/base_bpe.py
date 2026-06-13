# BPE 分词器共享基类 — 字符级 BPE 与字节级 BBPE 的公共逻辑

import re
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class BaseBPETokenizer:
    """BPE 分词器基类。

    字符级 BPE 和字节级 BBPE 的唯一差异在三个入口：
      - _preprocess_train / _preprocess_encode / _postprocess_decode
    除此以外，统计合并、编码循环、特殊 token 处理全部共享。
    """

    def __init__(self):
        self.vocab: Dict[int, str] = {}          # ID -> subword
        self.inverse_vocab: Dict[str, int] = {}  # subword -> ID
        self.merges: Dict[Tuple[str, str], int] = {}  # 合并规则 (rank)

        # 统一特殊 token ID 体系
        self.special_tokens = {
            "<|endoftext|>": 1000,
            "<think>": 1001,
            "</think>": 1002,
            "<UNK>": 1003,
        }
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}

        # GPT 风格预分词正则（共享）
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )
        escaped = [re.escape(k) for k in self.special_tokens]
        self.special_pattern = re.compile(f"({'|'.join(escaped)})")

    # ── 子类重写的钩子 ──────────────────────────────────────────────

    def _preprocess_train(self, text: str) -> str:
        """训练前预处理：BBPE 做字节→可见字符映射，字符级 BPE 不做。"""
        return text

    def _preprocess_encode(self, text: str) -> str:
        """编码前预处理：BBPE 做字节→可见字符映射，字符级 BPE 不做。"""
        return text

    def _postprocess_decode(self, text: str) -> str:
        """解码后还原：BBPE 做可见字符→字节→UTF-8，字符级 BPE 不做。"""
        return text

    def _init_vocab(self, chars: List[str]) -> int:
        """用给定的字符列表初始化词表，返回下一个可用的 ID。

        字符级 BPE：传入训练数据中所有唯一字符。
        字节级 BBPE：传入 byte_encoder 的 256 个可见字符。
        """
        for i, char in enumerate(chars):
            self.vocab[i] = char
            self.inverse_vocab[char] = i
        return len(self.vocab)

    def _on_unknown_symbol(self, sym: str) -> int:
        """遇到未知符号时的回退策略。字符级 BPE 回退到 <UNK>。"""
        return self.special_tokens["<UNK>"]

    # ── 公共核心算法 ────────────────────────────────────────────────

    def _get_stats(
        self, vocab_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """统计相邻 pair 的频率。"""
        pairs = defaultdict(int)
        for word_tuple, freq in vocab_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_tuple(
        self, word_tuple: Tuple[str, ...], pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        """在元组中合并指定 pair，比正则更高效。"""
        new_word = []
        i = 0
        first, second = pair
        while i < len(word_tuple):
            if (
                i < len(word_tuple) - 1
                and word_tuple[i] == first
                and word_tuple[i + 1] == second
            ):
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word)

    # ── 训练 ────────────────────────────────────────────────────────

    def train(self, text: str, num_merges: int):
        """训练 BPE 模型。"""
        # 1. 规范化 + 字节级预处理（BBPE 专有）
        text = unicodedata.normalize("NFC", text)
        text = self._preprocess_train(text)

        # 2. 预分词
        words = self.pat.findall(text)
        vocab_freqs = Counter(tuple(list(w)) for w in words)

        # 3. 初始化词表（由子类决定初始字符集）
        chars = sorted(set(ch for wt in vocab_freqs for ch in wt))
        current_id = self._init_vocab(chars)
        print(f"[*] 初始字符数：{current_id}")

        # 4. 核心合并循环
        for i in range(num_merges):
            pairs = self._get_stats(vocab_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = i

            new_symbol = best_pair[0] + best_pair[1]
            self.vocab[current_id] = new_symbol
            self.inverse_vocab[new_symbol] = current_id
            current_id += 1

            new_vocab_freqs = {}
            for word_tuple, freq in vocab_freqs.items():
                new_tuple = self._merge_tuple(word_tuple, best_pair)
                new_vocab_freqs[new_tuple] = freq
            vocab_freqs = new_vocab_freqs

            if (i + 1) % 100 == 0:
                print(f"[*] 已完成 {i + 1} 轮合并！")

        print(f"[*] 已完成 {num_merges} 轮合并，最终词表大小: {len(self.vocab)}")

    # ── 编码 / 解码 ─────────────────────────────────────────────────

    def _encode_chunk(self, text: str) -> List[int]:
        """对非特殊 token 的文本块编码。"""
        words = self.pat.findall(text)
        ids = []
        for word in words:
            symbols = list(word)
            # 贪心合并：每次选 rank 最小的 pair
            while len(symbols) > 1:
                pairs = [
                    (symbols[i], symbols[i + 1])
                    for i in range(len(symbols) - 1)
                ]
                best_pair = min(
                    pairs, key=lambda p: self.merges.get(p, float("inf"))
                )
                if best_pair not in self.merges:
                    break
                symbols = list(self._merge_tuple(tuple(symbols), best_pair))

            for sym in symbols:
                if sym in self.inverse_vocab:
                    ids.append(self.inverse_vocab[sym])
                else:
                    ids.append(self._on_unknown_symbol(sym))
        return ids

    def encode(self, text: str) -> List[int]:
        """文本 → ID 序列。"""
        text = self._preprocess_encode(text)
        chunks = self.special_pattern.split(text)
        final_ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                final_ids.append(self.special_tokens[chunk])
            elif chunk:
                final_ids.extend(self._encode_chunk(chunk))
        return final_ids

    def decode(self, ids: List[int]) -> str:
        """ID 序列 → 文本。"""
        parts = []
        for idx in ids:
            if idx in self.special_tokens_inv:
                parts.append(self.special_tokens_inv[idx])
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
        return self._postprocess_decode("".join(parts))
