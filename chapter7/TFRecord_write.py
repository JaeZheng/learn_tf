#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/3 14:55
# @File    : TFRecord_write.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets("F:\\learn_tf\\chapter5\\MNIST_data", one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
# 训练数据的分辨率
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址
filename = "./data/output.tfrecords"
# 创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    if index % 1000 == 0:
        print("index: ", index)
    # 将图像矩阵转化为一个字符串
    image_raw = images[index].tostring()
    # 将一个样本的信息都转为Example Protocol Buffer的格式存储
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))

    # 写入
    writer.write(example.SerializeToString())
writer.close()