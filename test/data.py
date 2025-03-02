import json
from tqdm import tqdm
from collections import defaultdict
import os

json_path = 'data/LCCD.json'

# with open(json_path, 'r', encoding='utf-8') as f:
#     datas = json.load(f)
#     print(len(datas))

#     freq_words = defaultdict(int)

#     max_len = 2000
#     # with open('data/LCCD_ALL.txt', 'w', encoding='utf-8') as f:
#     #     for data in tqdm(datas):
#     #         for line in data:
#     #             for char in line:
#     #                 if '\u4e00' <= char <= '\u9fff':
#     #                     freq_words[char] += 1
        
#         # # 按照频率排序
#         # freq_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)
#         # print(freq_words)

#     with open('data/LCCD_2K.txt', 'w', encoding='utf-8') as f:
#         for data in tqdm(datas):
#             s = ""
#             for line in data:
#                 # for char in line:
#                     # if '\u4e00' <= char <= '\u9fff':
#                         # if freq_words[char] > 10000:
#                 s += line
#             s.replace('\n', '')
#             f.write(s + '\n')
#             max_len -= 0
#             if max_len <= 0:
#                 break
            

import re

def clean_text(text):
    # 去除冗余信息
    text = re.sub(r'[\s\n]+', ' ', text)  # 去除多余的空格和换行
    text = re.sub(r'[／．]', '', text)  # 去除特殊字符
    text = re.sub(r'：', ':', text)  # 将全角冒号转换为半角
    text = re.sub(r'－', '-', text)  # 将全角破折号转换为半角
    text = re.sub(r'，', ',', text)  # 将全角逗号转换为半角
    text = re.sub(r'！', '!', text)  # 将全角感叹号转换为半角
    text = re.sub(r'（', '(', text)  # 将全角括号转换为半角
    text = re.sub(r'）', ')', text)  # 将全角括号转换为半角
    text = re.sub(r'[ ]+', ' ', text)  # 去除多余的空格
    text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
    text = re.sub(r'\n+', '\n', text)  # 去除多余的空行
    # # 去除任意数量的连续符号
    # text = re.sub(r'([.,!?\-=*])\1+', r'\1', text)  # 匹配任意数量的连续符号并替换为单个符号
    # 匹配非中文字符的连续重复符号
    text = re.sub(r'([^\u4e00-\u9fa5])\1+', r'\1', text)  # 匹配非中文字符的连续符号并替换为单个符号
    return text.strip()

txts = os.listdir("data")

for txt in txts:
    with open(f"data/re_{txt}", 'w', encoding='utf-8') as fre:
        with open(f"data/{txt}", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 清洗文本
                cleaned_text = clean_text(line)
                fre.writelines(cleaned_text)
