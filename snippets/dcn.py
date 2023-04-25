# Copyright 2022 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements `Cross` Layer, the cross layer in Deep & Cross Network (DCN)."""

from typing import Union, Text, Optional

import tensorflow as tf


class Cross(tf.keras.layers.Layer):
    """一层Cross Layer"""

    def __init__(self, ......):
        super(Cross, self).__init__(**kwargs)

        self._projection_dim = projection_dim  # 矩阵分解时采用的中间维度
        self._diag_scale = diag_scale # 非负小数，用于改善训练稳定性
        self._use_bias = use_bias
        ......

    def build(self, input_shape):  # 定义本层要优化的参数
        last_dim = input_shape[-1]  # 输入的维度

        # [d,r]的小矩阵，d是原始特征的长度，r就是这里的_projection_dim
        # r << d以提升模型的计算效率，一般取r=d/4
        self._dense_u = tf.keras.layers.Dense(self._projection_dim, use_bias=False, )
        # [r,d]的小矩阵
        self._dense_v = tf.keras.layers.Dense(last_dim, use_bias=self._use_bias,)

    def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """ x0与x计算一次交叉
        x0:   原始特征，一般是embedding layer的输出。一个[B,D]的矩阵
              B=batch_size，D是原始特征的长度
        x:    上一个Cross层的输出结果，形状也是[B,D]
        输出:  也是形状为[B,D]的矩阵 
        """
        if x is None:
            x = x0  # 针对第一层

        # 输出是x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
        # 其中.* 代表按位相乘,
        # W分解成两个小矩阵的乘积，W=U*V，以减少计算开销,
        # diag_scale非负小数，加到W的对角线上，以增强训练稳定性
        prod_output = self._dense_v(self._dense_u(x))

        if self._diag_scale:# 加大W的对角线，增强训练稳定性
            prod_output = prod_output + self._diag_scale * x

        return x0 * prod_output + x
