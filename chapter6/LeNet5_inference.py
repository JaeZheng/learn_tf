#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/1 11:12
# @File    : LeNet5_inference.py

"""
一个类似LeNet-5的卷积神经网络模型
"""


import tensorflow as tf

image_size = 28
num_of_channels = 1
num_of_labels = 10

input_node = 784
output_node = 10

# 第一层卷积层的尺寸和深度
conv1_deep = 32
conv1_size = 5
# 第二层卷积层的尺寸和深度
conv2_deep = 64
conv2_size = 5
# 全连接层的节点个数
fc_size = 512


# 获取神经网络变量的函数。
def get_weights_variable(shape):
    weights = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return weights


def get_biases_variable(shape, initial_num):
    biases = tf.get_variable("bias", shape, initializer=tf.constant_initializer(initial_num))
    return biases


# 卷积和池化
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义神经网络的前向传播过程
def inference(input_tensor, train, regularizer):
    # 第一层, 卷积层
    with tf.variable_scope("layer1-conv1"):
        # 卷积核shape为[卷积核高度, 卷积核深度, 通道数量(输入通道数), 卷积核深度(输出通道数)]
        conv1_weights = get_weights_variable([conv1_size, conv1_size, num_of_channels, conv1_deep])
        conv1_biases = get_biases_variable([conv1_deep], 0.0)
        # 卷积步长为1, padding使用全0补充
        conv1 = conv2d(input_tensor, conv1_weights)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 第二层, 池化层
    with tf.variable_scope("layer2-pool1"):
        # 2*2池化
        pool1 = max_pool_2x2(relu1)
    # 第三层，卷积层
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = get_weights_variable([conv2_size, conv2_size, conv1_deep, conv2_deep])
        conv2_biases = get_biases_variable([conv2_deep], 0.0)
        conv2 = conv2d(pool1, conv2_weights)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 第四层，池化层
    with tf.variable_scope("layer4-pool2"):
        pool2 = max_pool_2x2(relu2)
    # 第四层输出为7*7*64的矩阵，要转化为第五层全连接层的输入格式
    pool_shape = pool2.get_shape().as_list()  # 获取pool2层之后的数据尺寸
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])  # pool_shape[0]为一个batch中的数据个数
    # 第五层，全连接层
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = get_weights_variable([nodes, fc_size])
        fc1_biases = get_biases_variable([fc_size], 0.1)
        # 全连接层的参数要加入正则化
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)  # 如果是在训练过程中，使用dropout，随机不激活50%神经元
    # 第六层，全连接层, 即输出层
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = get_weights_variable([fc_size, num_of_labels])
        fc2_biases = get_biases_variable([num_of_labels], 0.1)
        # 全连接层的参数要加入正则化
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit



