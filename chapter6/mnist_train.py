#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/30 15:26
# @File    : mnist_train.py

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import datetime
# 加载mnist_inference.py中定义的常量和前向传播的函数
from chapter6 import mnist_inference

# 配置神经网络训练的超参数
batch_size = 100
learning_rate_base = 0.8  # 基础学习率
learning_rate_decay = 0.99  # 学习率的衰减率
regularization_rate = 0.0001  # 正则化项在损失函数中的系数
training_steps = 30000
moving_average_rate = 0.99  # 滑动平均衰减率

# 模型保存的路径和文件名
model_save_path = "./model/"
model_name = "mnist_model.ckpt"


# 训练过程函数
def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.image_size**2], name="x_input")
    reshaped_x = tf.reshape(x,
                            [-1,
                             mnist_inference.image_size,
                             mnist_inference.image_size,
                             mnist_inference.num_of_channels])
    y_true = tf.placeholder(tf.float32, [None, mnist_inference.num_of_labels], name="y_true")
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    # 计算前向传播
    y = mnist_inference.inference(reshaped_x, True, regularizer)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值。指定为不可训练(trainable=False)
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_rate, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。tf.trainable_variables()返回的就是所有trainable=True训练参数的集合
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_true, 1))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_true)
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 总的损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,   # 基础的学习率
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples/batch_size,  # 过完所有训练数据需要的迭代次数
        learning_rate_decay,  # 学习率衰减速度
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 一边更新参数，一边更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化tf持久化类
    saver = tf.train.Saver()

    # 初始化会话并开始训练过程
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config=session_conf) as sess:
        tf.global_variables_initializer().run()

        # 迭代训练神经网络
        for i in range(training_steps):
            # 生成batch训练数据
            xs, ys = mnist.train.next_batch(batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_true: ys})
            if i % 1000 == 0:
                print("After {:d} training step(s), loss on training batch is {:g}"
                      .format(step, loss_value))
                # 保存当前模型
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)


# 主程序入口
def main(argv=None):
    start_time = datetime.datetime.now()
    mnist = input_data.read_data_sets("F:\\learn_tf\\chapter5\\MNIST_data", one_hot=True)
    train(mnist)
    end_time = datetime.datetime.now()
    print("running time: {} s".format((end_time - start_time).seconds))


# tf.app.run()会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()





