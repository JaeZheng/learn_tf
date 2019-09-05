#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/5 23:16
# @File    : LSTM_train.py

""" 训练神经网络实现对函数sin(x)取值的预测 """

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")

hidden_size = 30  # LSTM隐藏节点的个数
num_layers = 2    # LSTM的层数

time_steps = 10   # RNN的训练序列长度
training_steps = 10000  # 训练轮数
batch_size = 32

training_examples = 10000
testing_examples = 1000
sample_gap = 0.1  # 采样间隔


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq) - time_steps):
        x.append([seq[i: i+time_steps]])
        y.append([seq[i+time_steps]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(x, y, istraining):
    # 使用多层的LSTM结构
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(num_layers)
    ])

    # 使用tf接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    # 最后一个时刻的输出结果
    output = outputs[:, -1, :]

    # 接一层全连接层并计算损失
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)  # 平方差损失函数
    if not istraining:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1
    )

    return predictions, loss, train_op


def train(sess, train_x, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))


def run_eval(sess, test_x, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        predicton, _, _ = lstm_model(x, [0.0], False)

    predictions = []
    labels = []
    for i in range(testing_examples):
        p, l = sess.run([predicton, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


# 用正弦函数生成训练和测试数据集合。
test_start = (training_steps + time_steps) * sample_gap
test_end = test_start + (testing_examples + time_steps) * sample_gap
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, training_steps + time_steps, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, testing_examples + time_steps, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)