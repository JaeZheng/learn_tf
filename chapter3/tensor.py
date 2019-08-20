#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/8/20 20:10
# @File    : tensor.py


import tensorflow as tf

# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")
print(result)  # 输出的是一个张量变量，而不是计算结果
# 会话中计算一个张量的值
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    result_add = sess.run(result)
    print(result_add)
    # 以下代码有相同功能，计算一个张量的取值
    print(result.eval(session=sess))
