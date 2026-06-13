# 字节级 BPE（BBPE）详解

> 关联文档：[BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) — 工业级 BPE 分词流水线全流程概览  
> 关联代码：[bbpe_tokenizer.py](bbpe_tokenizer.py) · [byte_to_unicode_map.py](byte_to_unicode_map.py) · [base_bpe.py](base_bpe.py)

字节级 BPE（Byte-level BPE，简称 **BBPE**）是现代大模型（如 GPT-2, GPT-3, GPT-4, Llama, Baichuan 等）标配的分词技术。本文档深入解析 BBPE 的核心逻辑、字节映射技巧，以及 Unicode/UTF-8 的底层关系。与 [BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) 互为补充：本文侧重「字节级编码的底层原理」，BPE_Pipeline_Notes.md 侧重「工业流水线的全流程」。

---

## 目录

- [Part I：BBPE 核心逻辑与实现流程](#part-ibppe-核心逻辑与实现流程)
- [Part II：Unicode 与 UTF-8——从"字典"到"编码方案"](#part-iiunicode-与-utf-8从字典到编码方案)
- [Part III：如何在字符级 BPE 基础上升级为 BBPE](#part-iii如何在字符级-bpe-基础上升级为-bbpe)
- [Part IV：BBPE 面试核心考点](#part-ivbbpe-面试核心考点)

---

## Part I：BBPE 核心逻辑与实现流程

理解 BBPE 的核心逻辑，只需要记住一句话：**它不再从"字符"开始合并，而是从"字节（Byte）"开始合并。**

### 1. 为什么需要字节级 BPE？（动机）

传统的字符级 BPE（Character-level BPE）有两个致命痛点：
1. **词表爆炸**：Unicode 字符有 14 万多个。如果初始词表包含所有字符，词表底层就太大了。
2. **OOV（Out-of-Vocabulary）问题**：无论你收集多少字符，总会有新的 Emoji、生僻字或特殊符号出现。一旦出现没见过的字符，模型就"卡壳"了。

**BBPE 的解决方案**：
任何文本（无论是中文、英文还是 Emoji）在计算机底层都是由 **字节（0-255）** 组成的。如果我们从这 256 个基础字节开始合并，那么：
- **初始词表只有 256 个**（非常小）。
- **永远不会有 OOV**：因为任何字符都能表示为若干字节的组合。

### 2. BBPE 的核心逻辑步骤

BBPE 的训练和推理过程与普通 BPE 极其相似，但在"输入端"和"预处理"上有特殊设计。

#### 第一步：字节映射（The Byte-to-Unicode Trick）

这是 OpenAI 在 GPT-2 中引入的一个非常巧妙的技巧。
- **问题**：直接对原始字节流进行操作会遇到麻烦。比如字节 `0x20`（空格）、`0x0A`（换行符）或一些不可见字符，直接交给正则表达式（Regex）处理时可能会导致分词逻辑混乱（某些正则引擎会忽略或特殊处理这些字符）。
- **解决**：将 0-255 这 256 个字节值，映射到 **256 个可见的 Unicode 字符**上。
  - 例如：字节 `32` (空格) 映射为字符 `'Ġ'`，字节 `10` (换行) 映射为 `'Ċ'`。
  - 这样，底层的字节流就变成了一个由"看起来很奇怪但清晰可见"的字符组成的字符串。

> 此映射函数在本项目中对应 [byte_to_unicode_map.py](byte_to_unicode_map.py) 中的 `bytes_to_unicode()`。

#### 第二步：预分词（Pre-tokenization）

使用正则表达式将长文本切分成小的单元（通常是单词或短语）。
- **正则示例**：`'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+`
- 这一步是为了防止 BPE 将不相关的部分强行合并（比如把一个单词的结尾和下一个单词的开头合并）。

> 正则的深度解析见 [BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) 的 Part III–V。

#### 第三步：训练（统计与合并）

1. **初始化**：将每个单词转换成映射后的"可见字符"序列。
   - 比如单词 `hello` → 转换为字节 `[104, 101, 108, 108, 111]` → 映射为可见字符序列。
2. **统计频率**：统计所有相邻字符对（Pairs）出现的次数。
3. **合并**：将频率最高的 Pair 合并成一个新的 Token，并记录合并规则。
4. **循环**：重复上述过程，直到达到预设的词表大小（如 50257 或 32000）。

#### 第四步：编码（Encoding）

1. 拿到一段文本，先转为 UTF-8 字节。
2. 将每个字节通过"映射表"转为对应的可见字符。
3. 应用训练好的合并规则（按 Rank 顺序）。
4. 最后将得到的子词转为对应的 ID。

---

## Part II：Unicode 与 UTF-8——从"字典"到"编码方案"

我们可以用一个简单的比喻来开场：

- **Unicode** 是"**字典**"：它给世界上所有的字符都分配了一个唯一的"**页码**"。
- **UTF-8** 是"**写下页码的方式**"：它规定了如何用 0 和 1（字节）把这个页码记录在纸上。

### 1. 什么是 Unicode？（逻辑上的"码位"）

Unicode 是一个国际标准，它的目标是把世界上所有的文字（汉字、英文、阿拉伯文、希腊文、甚至 Emoji 😂）都纳入一个统一的表。

- 每个字符在 Unicode 中都有一个唯一的编号，叫做 **Code Point（码位）**。
- 通常写成 `U+四位十六进制数`。
  - 字母 `A` 的码位是 `U+0041`。
  - 汉字 `中` 的码位是 `U+4E2D`。
  - 笑脸 `😂` 的码位是 `U+1F602`。
- **注意**：Unicode 只是一个逻辑上的映射表，它并没有规定在计算机里怎么存储这些编号。

### 2. 什么是 UTF-8？（物理上的"编码"）

如果直接存储 Unicode 码位（比如用 4 个字节存一个编号），会非常浪费空间（因为英文 A 只需要 1 个字节就能存下）。于是有了 **UTF-8**。

**UTF-8 是一种变长的编码方式**。它根据码位的大小，决定用 1 到 4 个字节来表示一个字符：

| 字节数 | 码位范围 (十六进制) | 字节结构 (二进制) | 适用范围 |
| :--- | :--- | :--- | :--- |
| **1 字节** | `0000 - 007F` | `0xxxxxxx` | 标准 ASCII (英文、数字) |
| **2 字节** | `0080 - 07FF` | `110xxxxx 10xxxxxx` | 拉丁文、希腊文等 |
| **3 字节** | `0800 - FFFF` | `1110xxxx 10xxxxxx 10xxxxxx` | **绝大多数汉字**、日韩文 |
| **4 字节** | `10000 - 10FFFF`| `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | Emoji、生僻字、古文字 |

### 3. 为什么"所有字符"都可以表示为 UTF-8 字节？

1. **覆盖范围广**：Unicode 标准目前的码位范围是从 `U+0000` 到 `U+10FFFF`（一共约 111 万个位置）。目前的 UTF-8 设计足以覆盖这所有的码位。
2. **无缝转换**：只要一个符号被收入了 Unicode 字典，它就一定有一个唯一的码位。只要有码位，UTF-8 就能通过上面的数学规则把它拆解成 1~4 个字节。
3. **字节的穷举性**：无论一个字符在 UTF-8 里占几个字节，每一个字节的值一定在 `0~255` 之间。

**这就是 BBPE 的威力所在：**
- **字符级 BPE**：面对的是无限可能的字符（Unicode 码位），如果模型没见过某个字符，就报错。
- **字节级 BPE**：面对的是 256 个可能的字节。无论是什么奇怪的字符，拆开来不过是几个 `0~255` 之间的数字组合。由于 **256 个基础字节一定都在词表里**，模型永远能把任何序列切碎并读进去。

### 4. 总结：二者的核心区别

| 维度 | Unicode | UTF-8 |
| :--- | :--- | :--- |
| **性质** | 字符集（Character Set） | 编码方案（Encoding Scheme） |
| **内容** | 字符 → 编号 | 编号 → 二进制/字节 |
| **长度** | 逻辑上的编号，不谈长度 | 物理上的存储，1-4 字节变长 |
| **联系** | 它是 UTF-8 的基础 | 它是 Unicode 的一种实现方式 |

### 5. 面试加分回答（针对 BBPE）

> "之所以 BBPE 选择 UTF-8 字节流作为输入，是因为 **Unicode 码位空间太大（超过 10 万个字符）**，如果以字符为单位，初始词表会非常稀疏且容易出现 OOV。
>
> 而 **UTF-8 是一种变长编码**，它将 Unicode 码位映射到了字节维度。虽然一个汉字在 UTF-8 下被拆成了 3 个字节，但 **字节的取值范围永远只有 256 个**。
>
> 通过在 256 个基础字节上进行 BPE 合并，我们既保留了处理任何字符的能力（解决了 OOV），又通过合并高频字节对，重新构建出了类似于'字符'或'词'的语义单元，在效率和通用性之间取得了平衡。"

**一句话总结：Unicode 解决了"是什么"的问题，UTF-8 解决了"怎么存"的问题，而 BBPE 利用了"怎么存"的固定边界（256 字节）来搞定"是什么"的无限可能。**

---

## Part III：如何在字符级 BPE 基础上升级为 BBPE

在本项目的实现中，[bbpe_tokenizer.py](bbpe_tokenizer.py) 通过继承 [base_bpe.py](base_bpe.py) 并重写 4 个钩子方法，仅用约 50 行额外代码就将字符级 BPE 升级为字节级 BBPE。

### 3.1 初始化 byte_encoder / byte_decoder

```python
self.byte_encoder = bytes_to_unicode()          # 0-255 → 可见 Unicode 字符
self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
```

### 3.2 修改 `train` — 训练前做字节映射

```python
text = unicodedata.normalize('NFC', text)       # 规范化必须在字节编码之前！
raw_bytes = text.encode('utf-8')                 # 原始文本 → UTF-8 字节
text = ''.join(self.byte_encoder[b] for b in raw_bytes)  # 字节 → 可见字符
```

关键细节：**NFC 规范化必须在 `encode('utf-8')` 之前做**。字节映射后的字符串是"伪字符串"（如 `ä½ å¥½`），其唯一目的是保护原始字节。如果对它做 Unicode 规范化，引擎可能合并/分解这些特殊字符，破坏字节与字符之间的一一对应关系，导致无法解码回原始字节。

### 3.3 修改 `encode` — 编码前做同样的字节映射

```python
# 先将原始输入文本编码为 utf-8
raw_bytes = text.encode('utf-8')
# 再把字节流映射为可见 Unicode 字符
text = ''.join(self.byte_encoder[b] for b in raw_bytes)
```

### 3.4 修改 `decode` — 解码后还原字节

```python
# 可见字符 → 字节值 → bytes 对象 → UTF-8 解码
raw_bytes = bytes([self.byte_decoder[char] for char in text])
return raw_bytes.decode('utf-8', errors='replace')
```

`errors='replace'` 用于处理残缺字节序列：如果字节流不是合法 UTF-8，用替换字符 `�` (U+FFFD) 代替，避免线上推理崩溃。三种策略的取舍：
- `errors='strict'`（默认）：直接抛出 `UnicodeDecodeError`，程序崩溃——在 LLM 线上推理中绝对不能接受
- `errors='ignore'`：直接把无法解释的字节扔掉，假装没看见——会丢失信息
- `errors='replace'`：替换为 `�` (U+FFFD)——最安全的回退策略

### 3.5 修改词表初始化 — 覆盖全部 256 个基础字节

```python
initial_chars = sorted(list(set(self.byte_encoder.values())))
for i, char in enumerate(initial_chars):
    self.vocab[i] = char
    self.inverse_vocab[char] = i
```

这是 BBPE 杜绝 OOV 的关键：无论训练数据中出现哪些字符，**初始词表一定包含全部 256 个基础字节**。任何一个新字符的 UTF-8 编码都逃不出这 256 的排列组合。

---

## Part IV：BBPE 面试核心考点

**Q：BBPE 和普通 BPE 的最大区别是什么？**

A：入口不同。普通 BPE 处理的是字符（Unicode code points），BBPE 处理的是字节（UTF-8 字节流）。BBPE 的初始词表大小固定为 256。

**Q：为什么要搞一个字节到 Unicode 的映射（Byte-to-Unicode）？**

A：直接处理字节流会遇到控制字符（如 `\n`）、空格等。这些字符在正则表达式中具有特殊含义，或者在打印输出时不可见。映射到一组可见的 Unicode 字符（如 `Ġ`, `Ċ`）可以让分词逻辑更稳健，且方便调试。

**Q：BBPE 真的能完全解决 OOV 吗？**

A：是的。因为任何数据（文本、图像、二进制）都可以表示为字节流。只要词表里包含了那 256 个基础字节，就没有任何序列是它无法编码的。

**Q：为什么 BBPE 效率高？**

A：虽然单条文本的 Token 序列可能会变长一点点（比如一个汉字拆成 3 个字节），但它极大地缩小了 Embedding 层的大小（初始词表只有 256），且避免了处理海量生僻字符的复杂性。同时词表从"无限"变为固定 256，Embedding 层参数量大幅缩小。

**Q：`train` 中 NFC 规范化为什么必须在字节编码之前？**

A：字节映射后的字符串是"伪字符串"，其唯一目的是保护原始字节不被正则引擎误处理。如果在映射后再做 NFC 规范化，Unicode 引擎可能会合并或分解这些特殊字符，破坏字节与字符的一一对应关系，导致 decode 无法还原。

**建议**：在手撕代码时，如果你能写出那个 `bytes_to_unicode` 的映射思想，面试官会认为你对 Transformer Tokenization 的底层细节有极深的掌握。

---

## 本目录文件索引

| 文件 | 说明 |
| :--- | :--- |
| [README.md](README.md) | 本目录的入门索引：文件清单 + 学习路径 + 运行指令 |
| [base_bpe.py](base_bpe.py) | BPE 共享基类（`_get_stats` / `_merge_tuple` / `_encode_chunk`） |
| [bpe_tokenizer.py](bpe_tokenizer.py) | 字符级 BPE 分词器（`BPETokenizer`），继承 base_bpe |
| [bbpe_tokenizer.py](bbpe_tokenizer.py) | 字节级 BBPE 分词器（`BBPETokenizer`），继承 base_bpe + byte_encoder |
| [byte_to_unicode_map.py](byte_to_unicode_map.py) | `bytes_to_unicode()` 工具函数，0–255 字节 → 可见 Unicode 字符 |
| [BPE_Pipeline_Notes.md](BPE_Pipeline_Notes.md) | 工业级 BPE 分词流水线的完整六步流程深度解析 |
| [BBPE.md](BBPE.md) | 本文档 —— 字节级 BPE 核心原理、Unicode/UTF-8 关系与面试考点 |
