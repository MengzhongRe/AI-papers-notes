# 手动实现大模型的词嵌入层(Word Embedding)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyEmbedding(nn.Module):
    def  __init__(self,vocab_size: int,embed_dim: int):
        super().__init__()
        # Embedding层的本质就是一个巨大的权重矩阵
        # [vocab_size,embed_dim]
        # nn.Parameter()函数包裹的torch.Tensor会自动设置requres_grad=True,并自动加入到模型的model.parameters()参数表中
        # 以便optimizer更新参数
        self.weight = nn.Parameter(torch.randn((vocab_size,embed_dim)))
    
    def forward_math_equivalent(self,input_ids):
        """
        写法一:数学等价:用Linear 和 matmul实现
        """
        # inputd_ids: [Batch_size,Seq_len]
        # 1.离散符号转为独热向量
        # one_hot_vectors: [Batch_size,Seq_len,vocab_size]
        # one_hot_vectors起初是整数张量，但由于self.weight是浮点数参数，矩阵乘法必须是浮点数和浮点数相乘
        # 由于矩阵乘法不触发pytorch的类型提升，Type Promotion，只有逐元素操作即 + - * /才会
        # 因此需要用float()转换
        one_hot_vectors = F.one_hot(input_ids,num_classes=self.weight.size(0)).float()

        # 2.矩阵乘法
        # [Batch_size,Seq_len,Vocab_size] @ [Vocab_size,embed_dim] -> [Batch_size,Seq_len,embed_dim]
        output = torch.matmul(one_hot_vectors,self.weight)

        return output

    def forward_engineering_real(self,input_ids):
        """
        写法二：工程实际
        """
        # input_ids: [Batch_size,Seq_len]

        # 利用pytorch的高级索引(advanced indexing)，直接把input_ids当作行号
        # 去weight矩阵里查表
        # 瞬间得到[Batch_size,Seq_len,embed_dim]

        output = self.weight[input_ids]
        return output


if __name__ == '__main__':
    V = 32000
    dim = 4096

    # 模拟输入
    input_ids = torch.tensor([[42,100,9999],[376,876,3]])
    my_embedding = MyEmbedding(V,dim)

    out_math = my_embedding.forward_math_equivalent(input_ids)
    out_real = my_embedding.forward_engineering_real(input_ids)

    assert torch.allclose(out_math,out_real,atol=1e-5),'结果不相等'

