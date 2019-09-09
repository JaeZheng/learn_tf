#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/8 17:15
# @File    : PTB_preprocess.py

"""按词频顺序建立词汇表，并保存到一个独立的vocab文件"""

import codecs
import collections
from operator import itemgetter

raw_data = "./data/ptb.train.txt"
vocab = "./data/ptb.vocab"

counter = collections.Counter()  # 统计单词出现频率
with codecs.open(raw_data, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1


# 按词频排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
# 加入句子结束符eos
sorted_words = ["<eos>"] + sorted_words

# 写入词汇文件
with codecs.open(vocab, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")


