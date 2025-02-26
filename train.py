import os
from io import open
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import math
from collections import defaultdict
import jieba

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path)

    def tokenize(self, paths):
        """Tokenizes a text file."""
        idss = []
        for path in paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                for line in tqdm(f):
                    ids = []
                    for word in line.replace(' ', '').replace('\n', ''):
                        self.dictionary.add_word(word)
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))

        ids = torch.cat(idss)

        return ids

# 定义一个位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
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
            self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # 多头自注意力层
        # query, key, value 都为 x
        attn_output, _ = self.self_attn(x, x, x, attn_mask=self.mask, is_causal=True, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
    
# 定义一个transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_encoder_layers, max_len, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=1.0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len)

        self.encoder = CustomTransformerEncoder(num_layers=num_encoder_layers, embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=embed_dim)

        self.fc1 = nn.Linear(embed_dim, vocab_size)

        self.mask = None

    # def weight_init(self):


    def generate_square_subsequent_mask(self, sz):
        """生成一个自回归掩码（遮挡未来的 token）"""
        mask = torch.triu(torch.ones(sz, sz).bool(), diagonal=1)
        return mask

    def forward(self, src):
        src = self.positional_encoding(self.embedding(src))

        # output, _ = self.encoder(src, src, src, need_weights=False, attn_mask=self.mask)
        output = self.encoder(src)
        output = self.fc1(output)
        # output = self.fc2(output)
        return output

    
class TextDataset(Dataset):
    def __init__(self, data, seq_len=128):
        super(TextDataset, self).__init__()
        self.seq_len = seq_len
        # nbatch = data.size(0) // seq_len
        # data = data.narrow(0, 0, nbatch * seq_len)
        # self.data = data.view(-1, seq_len).contiguous()
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len-1], self.data[idx+1:idx+self.seq_len]

def collate_fn(batch):
    batch = pad_sequence(batch, padding_value=0, batch_first=True)
    return batch[:, :-1], batch[:, 1:]  # 输入（去掉最后一个），输出（去掉第一个）


if __name__ == "__main__":

    corpus = Corpus(["data/santi.txt"])

    # 展示前10个单词
    # for i in range(10):
    #     print(corpus.dictionary.idx2word[corpus.train[i]])

    ntokens = len(corpus.dictionary)
    max_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = TransformerModel(vocab_size=ntokens, embed_dim=256, num_heads=2, num_encoder_layers=2, max_len=max_len, dim_feedforward=32, dropout=0.1)
    try:
        mode.load_state_dict(torch.load("runs/model.pth", weights_only=True), strict=False)
    except:
        pass
    # print(mode)
    model = mode.to(device)

    # 将数据集并行化
    # [124132] ==> [n * 128] 数据长度
    dataset = TextDataset(corpus.train, seq_len=max_len)

    # 生成数据加载器，8线程
    data_loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=True)

    # 定义优化器
    initial_lr = 0.0005
    optimizer = torch.optim.AdamW(mode.parameters(), lr=initial_lr)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 定义余弦退火学习率调度器
    max_epoch = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)
    try:
        # 训练模型
        max_iter = 1000000
        iteration = 0
        show_iter = 100
        gradient_accumulation_steps = 1
        for epoch in range(max_epoch):
            mode.train()
            total_loss = 0
            par = tqdm(data_loader)
            for i, (src, tgt) in enumerate(par):

                # # 展示这些文本 
                # for i in range(len(src)):
                #     print("".join([corpus.dictionary.idx2word[j] for j in src[i].numpy().tolist()]))
                #     print("".join([corpus.dictionary.idx2word[j] for j in tgt[i].numpy().tolist()]))

                iteration += 1
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = mode(src)
                # print(output)
                loss = loss_fn(output.view(-1, ntokens), tgt.view(-1))

                output = torch.argmax(output, dim=-1)
                text = ' '
                for j in output[0]:
                    text  = text + corpus.dictionary.idx2word[j] 
                if iteration % show_iter == 1:
                    print("".join([corpus.dictionary.idx2word[j] for j in tgt[0].cpu().numpy().tolist()]))
                    print(text)
                
                loss.backward()
                if iteration % gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(mode.parameters(), 0.25)
                    optimizer.step()
                    scheduler.step()
                total_loss += loss.item()

                par.set_description(f"Epoch: {epoch}, iteration: {iteration}/{max_iter}, lr: {scheduler.get_last_lr()[0]:.6f}, Loss: {total_loss / (i + 1):.6f}")
                
                # 展示一段文字的生成效果
                if iteration % show_iter == 1:
                    mode.eval()

                    with torch.no_grad():
                        test_text = "看"
                        # test_text = test_text.split(' ')
                        src = torch.tensor([corpus.dictionary.word2idx[i] for i in test_text]).unsqueeze(0).to(device)
                        src = src[:, -max_len:]
                        gen_txt = ""
                        for i in range(64):
                            output = mode(src)
                            # print(output[0])
                            output = torch.argmax(output, dim=-1)
                            if src.size(1) >= max_len:
                                src = torch.cat([src[:, 1:], output[:, -1].unsqueeze(0)], dim=-1)
                            else:
                                src = torch.cat([src, output[:, -1].unsqueeze(0)], dim=-1)
                            gen_txt += corpus.dictionary.idx2word[output[0, -1].cpu()]
                        print("----->", "".join(gen_txt))

                    mode.train()
            
                # scheduler.step()

            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr ** (1 - iteration / max_iter)

            torch.save(mode.state_dict(), "runs/model.pth")
        
    except KeyboardInterrupt:  
        torch.save(mode.state_dict(), "runs/model.pth")





