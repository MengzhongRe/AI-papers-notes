# RadixAttention 代码走读

> 关联文档：[RadixAttention/README.md](README.md) · [radix_attention_notes.md](radix_attention_notes.md)
> 关联代码：[radix_attention.py](radix_attention.py)
>
> 本文是 [radix_attention.py](radix_attention.py) 的逐行代码走读，按照面试手撕的节奏，从计划设计到核心逻辑拆解，逐步解析 RadixCacheTree 的匹配、插入分裂、LRU 驱逐三大核心操作的实现细节。文中内嵌的代码片段与 radix_attention.py 中的实现一一对应，推荐先阅读本走读理解设计思路，再对照 .py 源文件查看完整可运行代码。
>
> 理论基础请参考 [radix_attention_notes.md](radix_attention_notes.md)。

## 目录

1. [计划表：面试手撕的抽象与简化](#一-计划表)
2. [手撕代码实现：RadixAttention 前缀缓存](#二-手撕代码实现radixattention-前缀缓存)
3. [测试与运行逻辑](#三-测试与运行逻辑-面试官最爱看的-simulation)
4. [代码的核心逻辑](#代码的核心逻辑)

---

### 一、 计划表

在面试手撕时，时间有限（通常只有 20-30 分钟），你不可能实现一个完整的 SGLang 框架。你需要**对系统进行合理的抽象与简化**：

1. **解耦 Token 与 物理块 (Physical Blocks)**
   - *建议*：在 Radix Tree 中，Key 是 Token 序列，Value 是对应的 Physical Block ID 列表。面试中，可以假设 `1 Token = 1 Block` 或者传入对齐的 `tokens` 和 `blocks` 列表，这样能避开复杂的“块内部 Token 偏移”计算。
2. **明确“分裂 (Split)”逻辑（核心考点）**
   - *建议*：Radix Tree (严格来说是 Patricia Trie) 的难点在于**插入时发现共享前缀，需要对现有节点进行分裂**。面试官一定会重点看你的 `LCP (Longest Common Prefix)` 计算和节点 Split 逻辑。
3. **Reference Count 怎么维护？**
   - *建议*：很多候选人会把 Ref Count 加在路径的所有节点上，这是错的/复杂的。**最佳实践**：只在**叶子节点**（或当前请求停留在的最后节点）增加 Ref Count。
4. **LRU 驱逐策略 (Eviction) 的安全保证**
   - *建议*：只需遍历/维护 `ref_count == 0` 且 `没有子节点 (len(children) == 0)` 的**叶子节点**进行驱逐。因为如果一个父节点有正在运行的子节点，它绝对不能被驱逐。

---

### 二、 手撕代码实现：RadixAttention (前缀缓存)

> 📋 以下嵌入代码供走读说明。完整可运行版本（含 assert 验证的冒烟测试）请直接查看 [radix_attention.py](radix_attention.py)，本走读嵌入版本无断言，仅供理解设计思路。

这里我为你准备了一份高水准的面试手撕代码，包含四个部分：**节点定义、前缀匹配、插入与分裂、LRU驱逐**。

```python
import time

class RadixNode:
    def __init__(self, tokens, blocks):
        self.tokens = list(tokens)    # Token IDs序列
        self.blocks = list(blocks)    # 对应的物理块 ID 序列
        self.children = {}            # Key: first_token -> RadixNode
        self.parent = None            # 指向父节点，方便驱逐时解绑
        
        self.ref_count = 0            # 引用计数 (代表当前有多少个请求正在使用该节点作为结尾)
        self.last_access_time = time.time() # 用于 LRU 驱逐

    def __repr__(self):
        return f"Node(tokens={self.tokens}, ref={self.ref_count})"


class RadixCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.root = RadixNode([],[])
        self.root.ref_count = float('inf')  # 根节点永不驱逐

    def match_prefix(self, tokens):
        """
        匹配最长前缀
        返回: (已匹配的 tokens, 已匹配的 blocks)
        """
        node = self.root
        idx = 0
        matched_blocks =[]

        while idx < len(tokens):
            key = tokens[idx]
            if key not in node.children:
                break

            child = node.children[key]
            child_tokens = child.tokens

            # 计算最长公共前缀 (LCP)
            i = 0
            while i < len(child_tokens) and (idx + i) < len(tokens):
                if child_tokens[i] == tokens[idx + i]:
                    i += 1
                else:
                    break

            if i == len(child_tokens):
                # 1. 完美匹配该子节点，继续往下找
                idx += i
                matched_blocks.extend(child.blocks)
                node = child
                node.last_access_time = time.time() # 更新 LRU
            else:
                # 2. 部分匹配该子节点，无法继续深入
                idx += i
                matched_blocks.extend(child.blocks[:i])
                break

        return tokens[:idx], matched_blocks

    def insert(self, tokens, blocks):
        """
        将新生成的 tokens 和 blocks 插入 Radix Tree
        返回: 插入后序列对应的最终节点 (用于后续 inc_ref)
        """
        if not tokens:
            return self.root

        node = self.root
        idx = 0

        while idx < len(tokens):
            key = tokens[idx]

            if key not in node.children:
                # 场景 1: 完全没有分支，直接新建节点
                new_node = RadixNode(tokens[idx:], blocks[idx:])
                new_node.parent = node
                node.children[key] = new_node
                return new_node

            child = node.children[key]
            child_tokens = child.tokens

            # 寻找 LCP
            i = 0
            while i < len(child_tokens) and (idx + i) < len(tokens):
                if child_tokens[i] == tokens[idx + i]:
                    i += 1
                else:
                    break

            if i == len(child_tokens):
                # 场景 2: 完全涵盖现有节点，继续向下
                idx += i
                node = child
                node.last_access_time = time.time()
            else:
                # 场景 3: 部分匹配，需要【分裂 (Split) 节点】 (重点考点!)
                
                # 3.1 创建分裂出来的后缀节点 (继承原子节点的 children 和 ref_count)
                split_node = RadixNode(child.tokens[i:], child.blocks[i:])
                split_node.children = child.children
                split_node.parent = child
                split_node.ref_count = child.ref_count
                split_node.last_access_time = child.last_access_time
                
                # 更新 split_node 的子节点的 parent 指针
                for c in split_node.children.values():
                    c.parent = split_node

                # 3.2 截断原节点
                child.tokens = child.tokens[:i]
                child.blocks = child.blocks[:i]
                child.children = {split_node.tokens[0]: split_node}
                child.ref_count = 0  # 原节点现在变成了中间节点，归零
                child.last_access_time = time.time()

                # 3.3 插入新的序列分支
                if idx + i < len(tokens):
                    new_node = RadixNode(tokens[idx+i:], blocks[idx+i:])
                    new_node.parent = child
                    child.children[new_node.tokens[0]] = new_node
                    return new_node
                else:
                    return child

        return node

    def evict(self, needed_blocks):
        """
        LRU 驱逐策略
        条件: ref_count == 0 且 是叶子节点 (没有 children)
        """
        freed_blocks = 0
        
        while freed_blocks < needed_blocks:
            leaves =[]
            
            # DFS 寻找可驱逐的叶子节点
            # (面试优化点说明: 实际工程中这里会用 LRU Doubly Linked List 或 Priority Queue 做到 O(1))
            def dfs(n):
                if n.ref_count == 0 and len(n.children) == 0 and n is not self.root:
                    leaves.append(n)
                for c in n.children.values():
                    dfs(c)
            
            dfs(self.root)

            if not leaves:
                raise MemoryError("Cache 空间不足，且没有可驱逐的节点！")

            # 找出最久未使用的叶子节点
            oldest_leaf = min(leaves, key=lambda x: x.last_access_time)
            
            # 释放物理块
            freed_blocks += len(oldest_leaf.blocks)
            
            # 从父节点中删除自己
            parent = oldest_leaf.parent
            del parent.children[oldest_leaf.tokens[0]]
            
            # (面试加分项：可以提一句如果有必要，这里可以执行 Merge 操作将父节点与其唯一的子节点合并)
            
        return freed_blocks

    # --- 引用计数 API ---
    def inc_ref(self, node):
        node.ref_count += 1

    def dec_ref(self, node):
        node.ref_count = max(0, node.ref_count - 1)
```

---

### 三、 测试与运行逻辑 (面试官最爱看的 Simulation)

为了向面试官证明你的代码是对的，你可以写一段伪测试，模拟真实大模型多轮对话的缓存利用过程：

```python
if __name__ == "__main__":
    cache = RadixCache(capacity=100)

    # 1. 第一个请求 (System Prompt + User Query)
    tokens_req1 =[101, 102, 103, 104]
    blocks_req1 = [1, 2, 3, 4]
    
    # 模拟匹配 (此时缓存为空，匹配长度为0)
    matched_toks, matched_blks = cache.match_prefix(tokens_req1)
    print(f"Req1 匹配到了: {matched_toks}") #[]
    
    # 将完整的请求插入 Tree，并增加引用计数
    node_req1 = cache.insert(tokens_req1, blocks_req1)
    cache.inc_ref(node_req1)
    print(f"Req1 插入后，根节点状态: {cache.root.children}\n")

    # 2. 第二个请求 (相同的 System Prompt，不同的 User Query)
    tokens_req2 =[101, 102, 201, 202]
    blocks_req2 =[1, 2, 5, 6]  # 前两个 block 复用 1, 2
    
    # 模拟匹配 (会匹配到 101, 102)
    matched_toks2, matched_blks2 = cache.match_prefix(tokens_req2)
    print(f"Req2 匹配到了: {matched_toks2}") # [101, 102]
    
    # 将新请求插入 Tree (这里会触发 Radix Tree 的 Split 逻辑!)
    node_req2 = cache.insert(tokens_req2, blocks_req2)
    cache.inc_ref(node_req2)
    
    print(f"Req2 插入 (Split) 后:")
    print(f"共有前缀节点: {cache.root.children[101]}")
    print(f"分支1: {cache.root.children[101].children[103]}")
    print(f"分支2: {cache.root.children[101].children[201]}\n")

    # 3. Req1 执行完毕，释放引用计数
    cache.dec_ref(node_req1)
    print("Req1 完成，释放 Ref Count...")
    
    # 4. 内存不足，触发驱逐 (要求驱逐 2 个 block)
    print("触发 Evict 释放 2 个 block...")
    freed = cache.evict(2)
    print(f"成功驱逐了 {freed} 个 block。当前树的叶子情况:")
    print(cache.root.children[101].children)  # 此时 [103, 104] 已经被删除了
```


## 代码的核心逻辑

### 核心逻辑 1：节点里到底存了什么？(`RadixNode`)

在普通的前缀树（Trie）里，每个节点可能只存一个 Token（比如字符 `a` -> `p` -> `p` -> `l` -> `e`）。
但在大模型中，这样做树会太深，查找效率极低。所以 Radix Tree **把连续的单行道合并成了一个节点**。

你可以把 `RadixNode` 想象成一个**“包裹”**，里面装了：
1. `tokens`：一小段 Token 序列（比如 `[101, 102, 103]`）
2. `blocks`：这些 Token 在显存里对应的物理块号（比如 `[块A, 块B, 块C]`）
3. `children`：它的后续分支（字典结构）
4. `ref_count`：**有几个并发请求正在使用这个节点？**（引用计数，为 0 时代表空闲，可以被驱逐）

---

### 核心逻辑 2：如何查找缓存？(`match_prefix`)

假设当前树里只有一个长节点：
`Root` -> `[101, 102, 103, 104]` (对应块 `[A, B, C, D]`)

**新来了一个请求：`[101, 102, 201, 202]`**
代码的查找逻辑是：
1. 从 Root 开始往下找，遇到子节点 `[101, 102, 103, 104]`。
2. 开始逐个对比（计算最长公共前缀 LCP）：
   - `101` 匹配！
   - `102` 匹配！
   - `201` 和 `103` **不匹配**！
3. 查找结束。代码会返回：我帮你找到了前缀 `[101, 102]`，它们存在物理块 `[A, B]` 里。

> **面试官视角：** 这里考查的是数组的双指针遍历，比较简单，只要别越界就行。

---

### 核心逻辑 3：如何插入和“分裂”？(`insert`) ⭐️ 最难懂的部分

顺着上面那个例子。新请求匹配到了 `[101, 102]`，但是接下来它要生成 `201, 202`。
这时候，原本那个完整的节点 `[101, 102, 103, 104]` 就**必须被劈开（Split）**，变成一个分叉路口！

**代码中的场景 3（Split 逻辑）是这样做的：**

**第一步：创建一个“分裂后缀节点”**
把原节点后面不匹配的部分 `[103, 104]` 拿出来，单独做成一个新的子节点。并且把原节点的 `ref_count` 和它原本的子孙，都**过继**给这个新节点。

**第二步：截断原节点**
把原节点削短，只保留公共前缀 `[101, 102]`。然后把第一步建好的 `[103, 104]` 挂在它下面作为子分支。

**第三步：插入新的分支**
把新请求独有的 `[201, 202]` 做成一个新节点，也挂在 `[101, 102]` 下面。

**图形化变化过程：**

**【分裂前】**
```text
Root
 └── Node: [101, 102, 103, 104] (ref=1)
```
**【分裂后】**
```text
Root
 └── Node:[101, 102] (截断后变成父节点, ref=0)
      ├── 分支1 (旧):[103, 104] (继承ref=1)
      └── 分支2 (新): [201, 202] (等会外部调 inc_ref 加1)
```
> **面试官视角：** 这是整段代码的**灵魂**。为什么截断后的父节点 `ref=0`，而旧分支要继承 `ref=1`？因为原本那个正在跑的请求，用的是完整的 `101->104` 路径，它的“终点”现在变成了 `[103, 104]` 这个节点，所以引用计数必须跟着“终点”走。

---

### 核心逻辑 4：显存满了怎么办？(`evict`)

当大模型的 KV Cache 显存满了，我们就需要踢掉一些旧的缓存（LRU 算法）。

**踢人的规则是什么？（代码中的 `dfs` 寻找叶子节点）**
1. **必须是 `ref_count == 0`**：表示当前没有任何请求在生成这句话，是空闲的。
2. **必须是叶子节点（没有 `children`）**：为什么？看上面的图。假设你要踢掉 `[101, 102]` 这个父节点，那底下的 `[103, 104]` 和 `[201, 202]` 就成了没爹的野孩子，整棵树就断了！**所以只能从树的最末端（叶子）开始像剥洋葱一样一层层删。**

**代码的执行过程：**
1. 遍历整棵树，把所有符合上述两个条件的节点找出来（装进 `leaves` 列表）。
2. 比较它们的 `last_access_time`（最后访问时间）。
3. 找到时间最久远的那片叶子，把它从它爸爸的 `children` 字典里删掉。
4. 释放它的物理块。如果显存还不够，重复这个过程。

---

### 💡 总结：一句话记住这份代码的精髓

*   `match_prefix` 就是**找相同点**。
*   `insert` 中的 Split 就是**遇到分歧，就把前面的相同点截断作为父节点，把分歧点变成两个兄弟节点**。
*   `evict` 就是**从树枝的最末梢开始剪，谁最久没用剪谁，绝不能从树干中间剪**。

你可以对照着这 4 个核心逻辑，再回头看一遍代码。重点看 `insert` 方法里标注了 `3.1`, `3.2`, `3.3` 的代码块，那是整个面试中最核心的采分点！