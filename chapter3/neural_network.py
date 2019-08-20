#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/20 20:42
# @File    : neural_network.py

import tensorflow as tf
from numpy.random import RandomState  # Numpy生成模拟随机数剧集

batch_size = 8
learning_rate = 0.001

# 定义参数权重
w1= tf.Variable(tf.truncated_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.truncated_normal([3, 1], stddev=1, seed=1))

# 定义输入输出的placeholder
x = tf.placeholder(tf.float32, shape=[None, 2], name="x_input")
y_true = tf.placeholder(tf.float32, shape=[None, 1], name="y_input")

# 前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_true * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
with tf.Session(config=config) as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("训练之前的参数：")
    print(sess.run(w1))
    print(sess.run(w2))
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_true:Y[start:end]})
        if i % 500 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_true:Y})
            print("After {:d} steps, cross entropy on all data is {:f}".format(i, total_cross_entropy))
    print("训练之后的参数：")
    print(sess.run(w1))
    print(sess.run(w2))

