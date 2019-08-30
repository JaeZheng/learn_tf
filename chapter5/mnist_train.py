#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/30 15:26
# @File    : mnist_train.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

# mnist数据集相关的常数
input_node = 784  # 输入图片的像素数量，即输入层节点数量
output_node = 10  # 输出层节点数量，即图像类别数

# 网络超参数
layer1_node = 500  # 隐层节点数量
batch_size = 100
learning_rate_base = 0.8  # 基础学习率
learning_rate_decay = 0.99  # 学习率的衰减率
regularization_rate = 0.0001  # 正则化项在损失函数中的系数
training_steps = 30000
moving_average_rate = 0.99  # 滑动平均衰减率


# 前向传播函数，给定输入张量和所有参数，计算网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 如果没有提供滑动平均类，则直接使用当前参数的值
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类先计算变量的滑动平均值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练过程函数
def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name="x_input")
    y_true = tf.placeholder(tf.float32, [None, output_node], name="y_true")

    # 隐层参数
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1, name="w1"))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]), name="b1")

    # 输出层参数
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1, name="w2"))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node], name="b2"))

    # 计算前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值。指定为不可训练(trainable=False)
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_rate, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。tf.trainable_variables()返回的就是所有trainable=True训练参数的集合
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_true, 1))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_true)
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总的损失函数
    loss = cross_entropy_mean + regularization
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

    # 计算准确率
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config=session_conf) as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据
        validate_feed = {
            x: mnist.validation.images,
            y_true: mnist.validation.labels
        }

        # 准备测试数据
        test_feed = {
            x: mnist.test.images,
            y_true: mnist.test.labels
        }

        # 迭代训练神经网络
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After {:d} training step(s), validation accuracy using average_model is {:g}"
                      .format(i, validate_acc))

            # 生成batch训练数据
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_true: ys})

        # 训练结束后输出测试集结果
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After {:d} training step(s), test accuracy using average_model is {:g}"
              .format(training_steps, test_acc))


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





