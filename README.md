记录NLP相关学习

如有不合理之处，请大佬轻虐

关于NLP文本生成，采用Ddcoder-Only形式

---

step_one_tran.py是初次运行基于transformer文本生成模型，训练数据来着小说《三体》
注意词汇库是单独的文字
在mha（MultiheadAttention）中的casual mask可以有两种形式，一种是-inf，另一种是True，具体可以看官网相关描述
本文件用于玩具形式的文本生成体验，可以在简单的GPU环境下完成，1050ti（4G）

---


