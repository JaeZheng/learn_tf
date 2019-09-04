#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/4 14:10
# @File    : Coordinator.py

import tensorflow as tf
import numpy as np
import threading
import time


# 线程中运行的函数，每隔1秒判断是否需要停止并打印自己的ID
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:  # 随机停止所有线程
            print("Stopping from id: {:d}\n".format(worker_id))
            # 通知其他线程停止
            coord.request_stop()
        else:
            print("Working on id: {:d}\n".format(worker_id))
        time.sleep(1)


# 声明一个协调器
coord = tf.train.Coordinator()
# 创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]
# 启动所有线程
for t in threads:
    t.start()
# 等待所有线程退出
coord.join()