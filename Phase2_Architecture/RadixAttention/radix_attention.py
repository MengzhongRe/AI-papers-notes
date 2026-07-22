# 用Pytorch手撕RadixAttention的RadixCacheTree
# 导入解决前向引用导致的类型注解报错问题
# 必须在首行导入,才能生效,否则会报语法错误
from __future__ import annotations
import time


# ==============================================
# 模块一: RadixTree节点类
# ==============================================
class RadixNode():

    def __init__(self, tokens: list[int], blocks: list[int]):
        """
        RadixTree中的一个节点类的实现,类中的核心数据包含其存储的token ids及其
        物理块ids,同时实现引用计数方便LRU驱逐,同时用哈希表存储其所有的子节点
        参数:
            tokens: [101,102,...]节点存储的token ids列表
            blocks: [05,25,...] token ids对应存储的物理块ids,一般而言
            一个物理块对应若干个token ids,但是这里为了不计算指针偏移量,作简化处理
        """
        # 防御性编程:由于外部传进来的列表tokens,blocks与类内部共享同一
        # 物理地址,外部的改变也会影响内部的,因此用list实现类的私有化
        # 此外,如果外部传进来的参数是元组,迭代器之类的也会保证是列表结构
        self.tokens = list(tokens)  # 节点存储的token ids列表
        self.blocks = list(blocks)  # token ids对应的物理块ids
        self.children = {}  # 节点的子节点哈希表 first_token: RadixNode
        self.parent = None  # 指向父节点,方便驱逐时解绑

        self.ref_count = 0  # 引用计数
        self.last_access_time = time.time()

    def get_first_token(self) -> int:
        """
        返回当前节点的key,即首个token id
        """
        return self.tokens[0] if self.tokens else None
    
    def update_access_time(self):
        """
        更新节点访问时间
        """
        self.last_access_time = time.time()

    def add_child(self, child_node: RadixNode):
        """
        将节点加入到当前节点的子节点中
        """
        child_node.parent = self
        key = child_node.get_first_token()
        self.children[key] = child_node
    
    def remove_child(self, child_node: RadixNode):
        """
        将某一个子节点安全地从当前节点的孩子中删除
        """
        key = child_node.get_first_token()
        if key in self.children:
            del self.children[key]
            child_node.parent = None

    def match_length(self, tokens: list[int], offset: int = 0) -> int:
        """
        计算并返回当前节点与给定偏移量下的tokens的匹配度
        """
        i = 0
        while i < len(self.tokens) and (offset + i) < len(tokens):
            if self.tokens[i] == tokens[offset + i]:
                i += 1
            else:
                break
        return i

    def split(self, split_idx: int) -> RadixNode:
        """
        逻辑: 将当前节点从split_idx处分裂
        返回: 新分裂出来的后缀子节点
        """
        # 1.生成新后缀子节点
        suffix_node = RadixNode(self.tokens[split_idx:], self.blocks[split_idx:])
        suffix_node.children = self.children
        suffix_node.ref_count = self.ref_count
        suffix_node.update_access_time()    # 更新节点访问时间
        # 将原节点的父节点指向新的后缀节点
        for child in suffix_node.children.values():
            child.parent = suffix_node

        # 2.将原节点截断到split_idx
        self.tokens = self.tokens[:split_idx]
        self.blocks = self.blocks[:split_idx]
        # 由于原节点变成了中间节点,其子节点需要先归零
        self.children = {}
        # 由于原节点变成了父节点,即中间节点,引用计数归零
        self.ref_count = 0
        # 更新节点访问时间
        self.update_access_time()
        # 后缀节点成为了原节点的子节点,因此需要加入到原子节点的children中
        self.add_child(suffix_node)
        return suffix_node


    # 类的内部方法,重写该函数能够改变print(node)的返回,方便调试时打印数据
    def __repr__(self):
        return f"RadixNode(tokens={self.tokens}, ref={self.ref_count})"

# =============================================================================
# 模块二： RadixCacheTree实现
# =========================================================================
class RadixCacheTree:
    """
    缓存树数据结构管理类: 仅仅负责树的匹配(match),插入(insert),驱逐(evict)
    """
    def __init__(self, capacity: int):
        """
        初始化RadixCacheTree类
        """
        self.capacity = capacity
        # 初始化前缀树的根节点,tokens,blocks都为空
        self.root = RadixNode([],[])    
        # 为防止根节点被驱逐,其引用计数设为正无穷
        self.root.ref_count = float('inf')

    def match_prefix(self, tokens: list[int]) -> Tuple[list[int],list[int]]:
        """
        在该前缀树上匹配给定token ids的最大前缀
        返回匹配到的token ids列表以及block ids列表子集
        """
        node = self.root    # 当前准备匹配的节点
        idx = 0 # 目前开始匹配的tokens的索引,即idx之前的token都已经匹配成功
        matched_blocks = [] # 已经匹配到的block ids

        while idx < len(tokens):
            key = tokens[idx]   # 目前正在匹配的token id
            if key not in node.children:    # 如果当前搜索节点的所有子节点的开头都不是key
                break
            # 获取匹配到的child
            child = node.children[key]
            # 调用节点方法获取检索节点child与tokens[idx:]前缀的匹配度
            # lcp是最长公共前缀
            lcp_length = child.match_length(tokens,idx)

            # 无论是部分匹配还是完全匹配,都更新访问时间
            child.update_access_time()
            # 更新blocks
            matched_blocks.extend(child.blocks[:lcp_length])
            idx += lcp_length

            if lcp_length == len(child.tokens):
                # 完全匹配,进入下一层
                node = child
            else:
                # 不完全匹配,无法深入
                break
        
        return tokens[:idx],matched_blocks

    def insert(self, tokens: list[int], blocks: list[int]) -> RadixNode:
        """
        将本次请求生成的token ids以及block ids插入到前缀树对应的位置中去
        返回插入的叶子节点
        """
        # 如果tokens为空,直接返回根节点
        if not tokens:
            return self.root
        # 从根节点开始匹配、插入
        node = self.root
        idx = 0 # 表示正在匹配的tokens索引

        while idx < len(tokens):
            key = tokens[idx]
            #  情况1: tokens[idx:]在这里完全不匹配,也就是key不在node的孩子节点中
            # 此时需要建立新的节点,带有tokens[idx:],blocks[idx:],并将其加入到node的孩子中
            if key not in node.children:
                new_node = RadixNode(tokens[idx:], blocks[idx:])
                node.add_child(new_node)
                # 已经插入完毕,返回插入的新节点
                return new_node

            # 情况2: key在node的孩子节点中,获取匹配到的child节点
            # 此时,我们需要获取lcp_len即该节点与tokens[idx:]的最大前缀长度
            child = node.children[key]
            lcp_len = child.match_length(tokens,offset=idx)
            idx += lcp_len
            # 事实上,情况2又分为三种小情况
            # 1)完全匹配: child的整个tokens都对得上,进入下一层继续探索
            if lcp_len == len(child.tokens):
                node = child
                node.update_access_time()
            # 不完全匹配 -> 触发分类
            else:
                # 只要是不完全匹配必须节点从lcp_len处分裂
                child.split(lcp_len)
                # 2）如果当前还有剩余的tokens没有匹配上
                if idx < len(tokens):
                    # 如果有剩余建个新节点挂上去
                    new_node = RadixNode(tokens[idx:],blocks[idx:])
                    # 将该节点加入到child的孩子中去
                    child.add_child(new_node)
                    # 返回该新生成的叶子节点
                    return new_node
                # 3) 没多余的了(idx == len(tokens))
                # 说明新请求刚好到分裂点结束,直接返回分裂后的父节点
                else:
                    return child
        
        return node

    def _collect_evictable_leaves(self, node: RadixNode, leaves_list: list[RadixNode]) -> list[RadixNode]:
        """
        辅助函数：DFS 搜索 node 节点及其子节点中所有可驱逐的叶节点。
        可驱逐条件：引用计数为 0 且无子节点且非根节点。
        """
        if node.ref_count == 0 and not node.children and node is not self.root:
            leaves_list.append(node)
        # 递归遍历node的所有孩子进行搜索
        for child in node.children.values():
            self._collect_evictable_leaves(child,leaves_list)


    def evict(self, needed_blocks: int) -> int:
        """
        LRU驱逐: 根据给定的需要的内存块数量,驱逐最久未使用的叶节点并释放其内存块
        返回最终释放的内存块数量
        """
        freed_blocks = 0
        
        while freed_blocks < needed_blocks:
            leaves = [] # 列表用于收集所有当前可驱逐的叶节点
            # 调用DFS辅助函数搜索所有当前可驱逐的叶节点
            self._collect_evictable_leaves(self.root, leaves)
            # 如果列表为空,则说明当前没有可驱逐的叶节点,即无法释放显存
            # 报内存错误
            if not leaves:
                raise MemoryError('Cache 空间不足!且没有可驱逐的叶节点!')
            # 找出最久未使用的叶节点
            oldest_leaf = min(leaves, key=lambda x: x.last_access_time)
            # 累加释放的内存空间
            freed_blocks += len(oldest_leaf.blocks)
            # 将驱逐的叶节点从其父节点处安全地移除
            oldest_leaf.parent.remove_child(oldest_leaf)
        
        return freed_blocks
    
    def inc_ref(self, node: RadixNode) -> None:
        """
        增加node引用计数
        """
        node.ref_count += 1
    
    def dec_ref(self, node:RadixNode) -> None:
        """
        安全地减少node引用计数
        """
        node.ref_count = max(0, node.ref_count - 1)


# ============================================================
# 模块三: 测试用例（Test Case）
# =============================================================
def test_radix_cache():
    print(' 🚀 现在开始对手撕的RadixCacheTree(基缓存树)进行测试...')
    # 实例化缓存树
    radix_cache_tree = RadixCacheTree(capacity = 100)
    
     # 1. 第一个请求 (System Prompt + User Query)
    tokens_ref1 = [101,102,103,104]
    blocks_ref1 = [1,2,3,4]

    # 模拟匹配: 此时缓存为空,匹配长度为0
    matched_toks,matched_blks = radix_cache_tree.match_prefix(tokens_ref1)
    print(f'Ref 1 匹配到了: {matched_toks}')

    # ===== 断言: 第一次查询空缓存，应当匹配到0个token和0个block =====
    assert len(matched_toks) == 0, f"第一次查询空缓存应匹配0个token, 实际: {len(matched_toks)}"
    assert len(matched_blks) == 0, f"第一次查询空缓存应匹配0个block, 实际: {len(matched_blks)}"

    # 将完整的请求插入Tree,并增加引用计数
    node_ref1 = radix_cache_tree.insert(tokens_ref1,blocks_ref1)
    radix_cache_tree.inc_ref(node_ref1)
    print(f'第一次请求插入后根节点的状态: {radix_cache_tree.root.children}\n')

    # ===== 断言: 插入后根节点应包含键为101的子节点,且引用计数为1 =====
    assert 101 in radix_cache_tree.root.children, "根节点中应该包含子节点101"
    assert radix_cache_tree.root.children[101].ref_count == 1, \
        f"插入后引用计数应=1, 实际={radix_cache_tree.root.children[101].ref_count}"
    # ===== 断言: 插入节点的tokens和blocks应与输入一致 =====
    assert radix_cache_tree.root.children[101].tokens == tokens_ref1, \
        "插入节点的tokens应该与输入一致"
    assert radix_cache_tree.root.children[101].blocks == blocks_ref1, \
        "插入节点的blocks应该与输入一致"

    # 2. 第二个请求 (相同的 System Prompt，不同的 User Query)
    tokens_ref2 = [101,102,201,202]
    blocks_ref2 = [1,2,5,6]

    # 模拟匹配 (会匹配到 101, 102)
    matched_toks,matched_blks = radix_cache_tree.match_prefix(tokens_ref2)
    print(f'第二次请求匹配到了: {matched_toks}')

    # ===== 断言: 第二次请求应该匹配到共享前缀 [101, 102] 及对应的 blocks [1, 2] =====
    assert matched_toks == [101, 102], f"应该匹配到前缀[101,102], 实际: {matched_toks}"
    assert matched_blks == [1, 2], f"应该匹配到blocks[1,2], 实际: {matched_blks}"

    # 将新请求插入 Tree (这里会触发 Radix Tree 的 Split 逻辑!)
    node_ref2 = radix_cache_tree.insert(tokens_ref2,blocks_ref2)
    radix_cache_tree.inc_ref(node_ref2)

    print('Ref2 插入(split)之后')
    print(f'共有共享前缀: {radix_cache_tree.root.children[101]}')
    print(f'分支1: {radix_cache_tree.root.children[101].children[103]}')
    print(f'分支2: {radix_cache_tree.root.children[101].children[201]}\n')

    # ===== 断言: 分裂后树结构正确 —— 共享前缀节点有两个分支 =====
    shared_node = radix_cache_tree.root.children[101]
    assert len(shared_node.children) == 2, f"分裂后应有2个分支, 实际: {len(shared_node.children)}"
    assert 103 in shared_node.children, "分支1 (token 103) 应存在"
    assert 201 in shared_node.children, "分支2 (token 201) 应存在"
    # 共享前缀节点本身引用计数为0（中间节点）
    assert shared_node.ref_count == 0, f"共享前缀节点(中间节点)引用计数应为0, 实际: {shared_node.ref_count}"

    # Ref 1完成,释放引用计数
    radix_cache_tree.dec_ref(node_ref1.children[103])
    print(f'Ref 1 完成,释放Ref Count...')
    print(f'Node状态为: {node_ref1.children[103]}\n')

    # ===== 断言: 释放后引用计数归零 =====
    assert node_ref1.children[103].ref_count == 0, \
        f"释放后引用计数应为0, 实际: {node_ref1.children[103].ref_count}"

    # 内存不足:驱逐(需要两个blocks)
    print("触发 Evict 释放 2 个 block...")
    freed = radix_cache_tree.evict(2)
    print(f'成功驱逐了 {freed} 个blocks,当前叶节点情况: ')
    print(f'{radix_cache_tree.root.children[101].children}')    # 此时[103,104]已被驱逐

    # ===== 断言: 驱逐验证 —— 应释放2个block,且被驱逐的节点已移除 =====
    assert freed == 2, f"应该驱逐2个block, 实际: {freed}"
    assert 103 not in radix_cache_tree.root.children[101].children, \
        "被驱逐的节点(103)应该已从树中移除"
    assert 201 in radix_cache_tree.root.children[101].children, \
        "未被驱逐的节点(201)应该仍然存在"
    # 验证剩余节点的 blocks 正确
    remaining_node = radix_cache_tree.root.children[101].children[201]
    assert remaining_node.tokens == [201, 202], \
        f"剩余节点的tokens应为[201,202], 实际: {remaining_node.tokens}"

    print(f' 🎉🎉🎉 恭喜你!所有测试均已通过!')


if __name__ == '__main__':
    test_radix_cache()