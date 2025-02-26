# 第三节中将多个Attention手动cat不够高效
# 可以将(batch_size, seq_len, embed_dim)的矩阵直接拆成(batch_size, seq_len, num_heads, head_dim)
# 其中 embed_dim = num_heads * head_dim

import torch
import torch.nn as nn

class MutilHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = embed_dim ** -0.5 # 用来缩放，防止QK相乘之后数据太大

        self.linearQ = nn.Linear(embed_dim, embed_dim)
        self.linearK = nn.Linear(embed_dim, embed_dim)
        self.linearV = nn.Linear(embed_dim, embed_dim)


    
    def forward(self, xq, xk, xv, att_mask=None):
        batch_size, seq_len, embed_dim = xq.shape
        # 输入: (batch_size, seq_len, embed_dim)
        Q = self.linearQ(xq).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.linearQ(xk).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.linearQ(xv).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 交换维度，变为 (batch_size, num_heads, seq_len, head_dim)
        # 识得计算的每一个头数据为 （seq_len, head_dim）
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if att_mask is not None: # 假设这里是上三角矩阵，用来掩盖未来信息
            attn_scores.masked_fill_(att_mask == 1, float('-inf')) # softmax(-inf) = 0

        attn_probs = torch.softmax(attn_scores, dim=-1)  # 归一化得到权重

        attn_output = torch.matmul(attn_probs, V)

        # 交换维度并 reshape 回 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)


        return attn_output


if __name__ == '__main__':

    ha = MutilHeadAttention(embed_dim=128, num_heads=4)

    inp = torch.randn(2, 16, 128)

    mask = torch.triu(torch.ones(16, 16), diagonal=1)

    out = ha(inp, inp, inp, att_mask=mask)

    print(out)
    print(out.shape)