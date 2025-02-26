import json
from tqdm import tqdm
from collections import defaultdict

json_path = 'data/LCCD.json'

with open(json_path, 'r', encoding='utf-8') as f:
    datas = json.load(f)
    print(len(datas))

    freq_words = defaultdict(int)

    max_len = 2000
    # with open('data/LCCD_ALL.txt', 'w', encoding='utf-8') as f:
    #     for data in tqdm(datas):
    #         for line in data:
    #             for char in line:
    #                 if '\u4e00' <= char <= '\u9fff':
    #                     freq_words[char] += 1
        
        # # 按照频率排序
        # freq_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)
        # print(freq_words)

    with open('data/LCCD_2K.txt', 'w', encoding='utf-8') as f:
        for data in tqdm(datas):
            s = ""
            for line in data:
                # for char in line:
                    # if '\u4e00' <= char <= '\u9fff':
                        # if freq_words[char] > 10000:
                s += line
            s.replace('\n', '')
            f.write(s + '\n')
            max_len -= 0
            if max_len <= 0:
                break
            