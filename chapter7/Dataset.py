#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/4 16:45
# @File    : Dataset.py

import tensorflow as tf

# 从一个数组创建数据集
input_data = [1,2,3,3,4]
dataset = tf.data.Dataset.from_tensor_slices(input_data)  # 从张量构建数据集
# 定义一个遍历器用于遍历数据集
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
y = x**2
with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))


# 文本数据集
input_files = ["./data/1.txt", "./data/2.txt"]
dataset = tf.data.TextLineDataset(input_files)   # 获取格式为多行文本的文本文件数据
# 定义迭代器
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(x))


# 从TFRecord数据文件中读取
# 解析一个TFRecord的方法
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'pixels': tf.FixedLenFeature([], tf.int64),
        })
    return features['image_raw'], features['label']


input_files = ['./data/output.tfrecords']
dataset = tf.data.TFRecordDataset(input_files)
# map()函数表示对数据集中的每一条数据调用相应方法
dataset = dataset.map(parser)
# 定义迭代器
iterator = dataset.make_one_shot_iterator()
f1, f2 = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        print(sess.run([f1, f2]))

# 具体文件路径是一个placeholder，后面再提供具体路径，这个时候迭代器初始化要使用make_initializable_iterator()方法
input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)
# 定义迭代器
iterator = dataset.make_initializable_iterator()
f1, f2 = iterator.get_next()
with tf.Session() as sess:
    # 首先初始化迭代器
    sess.run(iterator.initializer, feed_dict={input_files: ['./data/output.tfrecords']})
    for i in range(10):
        print(sess.run([f1, f2]))


