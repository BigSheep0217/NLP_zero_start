# import torch
# import torch.nn.functional as F
# seq_len = 5

# # 创建一个上三角矩阵（对角线以上设为1）
# future_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)

# # 将1的部分替换为 -inf
# future_mask = future_mask.masked_fill(future_mask == 1, float('-inf'))
# # 模拟 Q, K, V 矩阵 (batch_size=1, seq_len=5, d_model=4)
# batch_size, seq_len, d_model = 1, 5, 4
# Q = torch.rand(batch_size, seq_len, d_model)
# K = torch.rand(batch_size, seq_len, d_model)
# V = torch.rand(batch_size, seq_len, d_model)

# # 计算注意力分数 (QK^T)
# scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)

# # 应用未来遮罩
# scores = scores + future_mask.unsqueeze(0)  # 增加 batch 维度

# # 通过 softmax 归一化
# attention_weights = F.softmax(scores, dim=-1)

# # 计算加权 V
# output = torch.matmul(attention_weights, V)

# print(attention_weights)  # 观察未来的 token 是否被屏蔽



# def is_mostly_chinese(text, threshold=0.5):
#     total_chars = len(text)
#     if total_chars == 0:
#         return False
#     # 使用列表推导统计中文字符个数
#     s = ""
#     for char in text:
#         if '\u4e00' <= char <= '\u9fff':
#             s += char
#     return s

# # 示例
# text = "Hello, 这是一个测试。"
# print(is_mostly_chinese(text))  # 根据设定阈值判断
