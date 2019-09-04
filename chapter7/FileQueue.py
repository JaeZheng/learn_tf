#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/4 14:55
# @File    : FileQueue.py

import tensorflow as tf


# TFRecord帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下将数据写入不同的文件。
num_shards = 2  # 总共写入多少个文件
instance_per_shards = 2  # 每个文件中有多少个数据

for i in range(num_shards):
    file_name = "./data/tfrecords-%.5d-%.5d" % (i, num_shards)
    writer = tf.python_io.TFRecordWriter(file_name)
    for j in range(instance_per_shards):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j),
        }))
        writer.write(example.SerializeToString())
    writer.close()