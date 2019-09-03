#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/3 17:47
# @File    : picture_process.py

import matplotlib.pyplot as plt
import tensorflow as tf


image_raw_data = tf.gfile.FastGFile("./data/pic.jpg", 'rb').read()


def image_encode():
    # 图像编码处理
    # 解码读入图像
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    # 可视化图像
    plt.imshow(img_data.eval())
    plt.show()
    # 编码后写入图像
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.FastGFile("./data/output.jpeg", "wb") as f:
        f.write(encoded_image.eval())


def image_resize():
    # 图像大小调整
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    resized = tf.image.resize_images(img_data, [300, 300], method=0)  # method标识不同的图像大小调整算法
    plt.imshow(resized.eval())
    plt.show()
    # 原图像尺寸大于目标尺寸，自动截取居中部分
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 200, 200)
    plt.imshow(croped.eval())
    plt.show()
    # 原图像尺寸小于目标尺寸，四周填充0补全
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
    plt.imshow(padded.eval())
    plt.show()
    # 按比例裁剪图像
    central_croped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_croped.eval())
    plt.show()


def image_flip():
    # 图像翻转
    img_data = tf.image.decode_jpeg(image_raw_data)
    flipped = tf.image.flip_up_down(img_data)  # 上下翻转
    flipped = tf.image.flip_left_right(img_data)  # 左右翻转
    # flipped = tf.image.random_flip_up_down(img_data)  # 随机上下翻转
    # flipped = tf.image.random_flip_left_right(img_data)  # 随机左右翻转
    plt.imshow(flipped.eval())
    plt.show()
    transposed = tf.image.transpose_image(img_data)  # 对角线翻转
    plt.imshow(transposed.eval())
    plt.show()


def image_color():
    # 图像色彩调整
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)  # 图像亮度-0.5
    # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)  # 截断在一定范围内
    # plt.imshow(adjusted.eval())
    # plt.show()
    # adjusted = tf.image.adjust_brightness(img_data, 0.5)  # 图像亮度+0.5
    # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)  # 截断在一定范围内
    # plt.imshow(adjusted.eval())
    # plt.show()
    # # 在[-max_delta, max_delta]范围内随机调整图像的亮度
    # max_delta = 0.2
    # adjusted = tf.image.random_brightness(img_data, max_delta)
    # plt.imshow(adjusted.eval())
    # plt.show()
    # 将图像对比度减少到0.5倍
    # adjusted = tf.image.adjust_contrast(img_data, 0.5)
    # 将图像的对比度增加5倍
    adjusted = tf.image.adjust_contrast(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()


with tf.Session() as sess:
    image_color()