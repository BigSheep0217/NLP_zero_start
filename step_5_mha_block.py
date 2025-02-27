# 本节将mha加上ffn（前馈网络）组合成一个完整的Encoder Block

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


    
    def forward(self, xq, xk, xv, attn_mask=None):
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

        if attn_mask is not None: # 假设这里是上三角矩阵，用来掩盖未来信息
            attn_scores.masked_fill_(attn_mask == 1, float('-inf')) # softmax(-inf) = 0

        attn_probs = torch.softmax(attn_scores, dim=-1)  # 归一化得到权重

        attn_output = torch.matmul(attn_probs, V)

        # 交换维度并 reshape 回 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)


        return attn_output





class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        """
        :param embed_dim: 输入嵌入维度
        :param num_heads: 注意力头数
        :param ff_hidden_dim: 前馈网络隐藏层维度
        :param dropout: dropout 概率
        :param is_causal: 是否使用因果 mask (防止未来信息泄露)
        """
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.mask = None
        
        # 多头注意力层
        self.self_attn = MutilHeadAttention(embed_dim, num_heads)
        # self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, attn_mask=None):
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        :param attn_mask: 可选的注意力 mask，形状应为 (seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # 如果设置了因果 mask，并且没有传入 attn_mask，则自动构造上三角 mask
        if self.mask is None or self.mask.size(0) != seq_len:
            # 构造上三角 mask，mask 中的 True 表示要屏蔽的部分
            # 注意：nn.MultiheadAttention 接受的 attn_mask 为浮点数，其中屏蔽位置需要是 -inf
            self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
        
        # 多头自注意力层
        # query, key, value 都为 x
        attn_output, _ = self.self_attn(x, x, x, attn_mask=self.mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x
    
if __name__ == '__main__':

    ha = TransformerBlock(embed_dim=128, num_heads=4, ff_hidden_dim=512, dropout=0.1)

    inp = torch.randn(2, 16, 128)

    out = ha(inp)

    print(out)
    print(out.shape)