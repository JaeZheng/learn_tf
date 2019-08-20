#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/20 20:00
# @File    : tensor.py

import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 计算图g1中定义变量v，初始值为0
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    # 计算图g2中定义变量v,初始值为1
    v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer())

# 计算图g1中读取变量v的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()  # 全局初始化
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# 计算图g2中读取变量v的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()  # 全局初始化
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))