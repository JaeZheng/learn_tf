#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/30 10:49
# @File    : load_MNIST.py

from tensorflow.examples.tutorials.mnist import input_data

# 载入mnist数据集，如果不在指定路径，会自动下载
mnist = input_data.read_data_sets("F:\learn_tf\chapter5\MNIST_data", one_hot=True)

# 查看数据
print("Training data size: ", mnist.train.num_examples)
print("Validation data size: ", mnist.validation.num_examples)
print("Test data size: ", mnist.test.num_examples)
print("Example training data: ", mnist.train.images[0])
print("Example training label: ", mnist.train.labels[0])

# 批量提取数据
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size=batch_size)
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)

