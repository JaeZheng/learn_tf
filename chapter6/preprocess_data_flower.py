#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/2 15:26
# @File    : preprocess_data_flower.py

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

input_data = "F:\\learn_tf\\chapter6\\flower_data"
# 将处理后的图片数据通过numpy的格式保存
output_file = "./flower_data/flower_preprocess_data.npy"

validation_percentage = 10
test_percentage = 10


# 读取数据并分训练集、测试集、验证集
def create_image_lists(sess, testing_percentage, valiation_percentage):
    sub_dirs = [x[0] for x in os.walk(input_data)]
    print("sub_dirs ", sub_dirs)
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取一个子目录下所有图片文件
        extensions = ['jpg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        print("dir_name ", dir_name)
        for extension in extensions:
            file_glob = os.path.join(input_data, dir_name, "*."+extension)
            print("file_glob ", file_glob)
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
            print("len(file_list) ", len(file_list))

            # 处理图片数据
            for file_name in file_list:
                # 读取并解析图片，转为299*299以便inception-v3模型来处理
                image_raw_data = gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)

                # 随机划分数据集
                chance = np.random.randint(100)
                if chance < valiation_percentage:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif chance < (testing_percentage+valiation_percentage):
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)
            current_label += 1
    # 打乱训练数据
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(testing_labels)

    return np.asarray([training_images, training_labels, validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, test_percentage, validation_percentage)
        np.save(output_file, processed_data)


if __name__ == "__main__":
    main()