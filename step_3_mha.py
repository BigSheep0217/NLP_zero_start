# 本文在attention的基础上，拓展到多头注意力

import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    def __init__(self, embed_dim, laten_dim) -> None:
        super().__init__()
        self.scale = laten_dim ** -0.5 # 用来缩放，防止QK相乘之后数据太大
        self.linearQ = nn.Linear(embed_dim, laten_dim)
        self.linearK = nn.Linear(embed_dim, laten_dim)
        self.linearV = nn.Linear(embed_dim, laten_dim)

        self.laten_to_dmbed = nn.Linear(laten_dim, embed_dim) # 我们需要输出和输入保持一致，所以要映射回去
    
    def forward(self, xq, xk, xv, att_mask=None):
        # 输入: (batch_size, seq_len, embed_dim)
        Q = self.linearQ(xq)
        K = self.linearQ(xk)
        V = self.linearQ(xv)

        # Q和K用于计算元素之间的相关性，因为是生成任务，所以需要掩盖未来的数据
        # 如果是翻译这种全局数据已知的情况下，这里如果作为encoder的话不需要casual mask
        # V可以理解为将这些相关性都综合起来，类似于加权，而QK就是权重

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if att_mask is not None: # 假设这里是上三角矩阵，用来掩盖未来信息
            attn_scores.masked_fill_(att_mask == 1, float('-inf')) # softmax(-inf) = 0
        attn_probs = torch.softmax(attn_scores, dim=-1)  # 归一化得到权重
        attn_output = torch.matmul(attn_probs, V)
        # attn_output = self.laten_to_dmbed(attn_output)

        return attn_output


class MutilHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head, laten_dim=0) -> None:
        super().__init__()

        self.num_head = num_head
        self.head_dim = embed_dim // num_head # laten_dim类似于这个可以减少计算量，压缩通道数，但这里可以使用 // num_head

        self.att_heads = [HeadAttention(embed_dim, self.head_dim) for _ in range(self.num_head)]


    def forward(self, x, att_mask):
        
        atts = []
        for i in range(self.num_head):
            single_head = self.att_heads[i](x, x, x, att_mask)
            atts.append(single_head)

        out = torch.cat(atts, dim=-1)

        return out

if __name__ == '__main__':

    ha = MutilHeadAttention(embed_dim=128, num_head=4)

    inp = torch.randn(2, 16, 128)

    mask = torch.triu(torch.ones(16, 16), diagonal=1)

    out = ha(inp, att_mask=mask)

    print(out)
    print(out.shape)