"""
Author: Jiang Mingyang
email: jmydurant@sjtu.edu.cn
pointSIFT module op, do not modify it !!!
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

pointSIFT_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_pointSIFT_so.so'))

def pointSIFT_select(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius: float
    :return: (b, n, 8) int
    """
    idx = pointSIFT_module.cube_select(xyz, radius)
    return idx


ops.NoGradient('CubeSelect')

def pointSIFT_select_two(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 16) int
    """
    idx = pointSIFT_module.cube_select_two(xyz, radius)
    return idx


ops.NoGradient('CubeSelectTwo')

def pointSIFT_select_four(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 32) int
    """
    idx = pointSIFT_module.cube_select_four(xyz, radius)
    return idx


ops.NoGradient('CubeSelectFour')

