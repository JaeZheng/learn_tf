#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/5 22:34
# @File    : LSTM_demo.py

import tensorflow as tf

lstm_hidden_size = 512  # 隐状态向量的维度
batch_size = 64
num_steps = 300  # 序列长度

# 定义一个LSTM结构。tf中一句话就可以声明
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 将LSTM中的状态初始化为全0数组
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0

for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variables()

    lstm_output, state = lstm(state)
    # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
    # final_output = fully_connected(lstm_output)
    # 计算当前时刻的输出的损失
    # loss += cal_loss(final_output, expected_output)