#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/1 11:12
# @File    : mnist_inference.py

import tensorflow as tf

input_node = 784
output_node = 10
layer1_node = 500


# 获取神经网络变量的函数。regularizer可以将变量的正则化损失也加入损失集合
def get_weights_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层
    with tf.variable_scope("layer1"):
        weights = get_weights_variable([input_node, layer1_node], regularizer)
        biases = tf.get_variable("biases", [layer1_node], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 声明第二层
    with tf.variable_scope("layer2"):
        weights = get_weights_variable([layer1_node, output_node], regularizer)
        biases = tf.get_variable("biases", [output_node], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

