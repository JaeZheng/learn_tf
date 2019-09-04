#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/4 14:27
# @File    : QueueRunner.py

import tensorflow as tf

# 声明一个FIFO队列，最多100个元素，类型为实数
queue = tf.FIFOQueue(100, "float")
# 定义入队操纵
enqueue_op = queue.enqueue([tf.random_normal([1])])
# 使用QueueRunner来创建多个线程运行队列的入队操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)  # 启动五个线程，每个线程中运行的是enqueue_op操作
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # 启动所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(5):
        print(sess.run(out_tensor)[0])
    # 停止所有线程
    coord.request_stop()
    coord.join(threads)