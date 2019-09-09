#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/8 17:24
# @File    : vocab_num.py

"""将训练文件转化为单词编号的文件"""

import codecs
import sys

raw_data = "./data/ptb.valid.txt"
vocab = "./data/ptb.vocab"
output_data = "./data/ptb.valid"

with codecs.open(vocab, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# 如果出现了被删除的低频词，则替换为"<unk>"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


fin = codecs.open(raw_data, 'r', 'utf-8')
fout = codecs.open(output_data, 'w', 'utf-8')

for line in fin:
    words = line.strip().split() + ['<eos>']
    out_line = ' '.join([str(get_id(w)) for w in words]) + "\n"
    fout.write(out_line)

fin.close()
fout.close()