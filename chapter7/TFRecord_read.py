#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/3 15:22
# @File    : TFRecord_read.py

import tensorflow as tf

# 创建一个reader来读取TFRecord
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["./data/output.tfrecords"])

# 读取一个样例
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'pixels': tf.FixedLenFeature([], tf.int64),
    })

# tf.decode_raw可以把字符串解析成图像对应的像素数组
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可读取TFRecord文件中的一个样例
for i in range(10):
    print(sess.run([image, label, pixels]))