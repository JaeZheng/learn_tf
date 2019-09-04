#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/4 15:48
# @File    : batching.py

import tensorflow as tf

# tf.train.match_filenames_once获取文件列表
files = tf.train.match_filenames_once("./data/tfrecords-*")
# tf.train.string_input_producer创建输入文件的列表，shuffle设置是否打乱
file_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })

example, label = features['i'], features['j']

batch_size = 5  # 一个batch中的样例个数
capacity = 1000 + 3*batch_size  # 组合样例的队列中最多可以存储的样例个数

# tf.train.batch函数用于组合样例
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 获取并打印组合之后的样例
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)

