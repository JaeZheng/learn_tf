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

input_data = "./flower_data"
output_file = "./flower_data/flower_preprocess_data.npy"