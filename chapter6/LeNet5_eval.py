#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/1 11:39
# @File    : LeNet5_eval.py

import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from chapter6 import LeNet5_inference, LeNet5_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
eval_interval_secs = 10


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, LeNet5_inference.input_node], name="x_input")
    y_true = tf.placeholder(tf.float32, [None, LeNet5_inference.output_node], name="y_true")
    validate_feed = {x: mnist.validation.images, y_true: mnist.validation.labels}
    # 计算前向传播，因为是测试，不需关注正则化的值
    reshaped_x = tf.reshape(x,
                            [-1,
                             LeNet5_inference.image_size,
                             LeNet5_inference.image_size,
                             LeNet5_inference.num_of_channels])
    y = LeNet5_inference.inference(reshaped_x, False, None)
    # 计算准确率
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 通过变量重命名的方式来加载模型，这样在前向传播中就不需要调用求滑动平均值的函数来获取平均值了
    variable_averages = tf.train.ExponentialMovingAverage(LeNet5_train.moving_average_rate)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        # 初始化会话并开始训练过程
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        with tf.Session(config=session_conf) as sess:
            # 找到目录中最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(LeNet5_train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After {:s} training step(s), validation accuracy = {:g}".format(global_step, accuracy_score))
            else:
                print("No checkpoint file found.")
                return
        time.sleep(eval_interval_secs)


def main(argv=None):
    mnist = input_data.read_data_sets("F:\\learn_tf\\chapter5\\MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()