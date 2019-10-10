#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 上午11:16
# @Author  : Zessay

import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm

class FFM(object):
    def __init__(self, hparams, df_i, df_v):
        '''
        :param hparams: 是nametuple或者类对象，表示需要用到的参数
        :param df_i: [batch, n]，表示一个batch中需要用到的特征的索引
        :param df_v: [batch, n]，表示一个batch中需要用到的特征对应的值
        '''
        self.hparams = hparams
        tf.set_random_seed(self.hparams.seed)
        self.line_result = self.line_section(df_i, df_v)
        self.ffm_result = self.ffm_section(df_i, df_v)
        self.logits = self.line_result + self.ffm_result


    def line_section(self, df_i, df_v):
        with tf.variable_scope("line"):
            # 一阶特征权重，维度为 [features, 1]
            weights = tf.get_variable("weights",
                                      shape=[self.hparams.feature_nums, 1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.glorot_uniform())
            batch_weights = tf.nn.embedding_lookup(weights, df_i)
            batch_weights = tf.squeeze(batch_weights, axis=2)
            line_result = tf.multiply(df_v, batch_weights, name="line_w_x")
            bias = tf.get_variable("bias", shape=[1,1],
                                   dtype=tf.float32, initializer=tf.initializers.zeros())
            line_result = tf.add(tf.reduce_sum(line_result, axis=1, keepdims=True), bias)
        return line_result

    def ffm_section(self, df_i, df_v):
        with tf.variable_scope("ffm"):
            # 二阶特征权重，维度为 [field_size, features, embedding_size]
            embedding = tf.get_variable("embedding",
                                        shape=[self.hparams.field_nums,
                                               self.hparams.feature_nums,
                                               self.hparams.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.random_normal())
            ffm_result = None
            for i in range(self.hparams.field_nums):
                for j in range(i+1, self.hparams.field_nums):
                    ## 得到场i对于场j的隐向量, [batch, embedding_size]
                    ## 每一列对应相同的场
                    vi_fj = tf.nn.embedding_lookup(embedding[j], df_i[:, i])
                    ## 得到场j对于场i的隐向量，[batch, embedding_size]
                    vj_fi = tf.nn.embedding_lookup(embedding[i], df_i[:, j])
                    wij = tf.multiply(vi_fj, vj_fi)  # [batch, embedding_size]

                    ## 维度 [batch, 1]
                    x_i = tf.expand_dims(df_v[:, i], 1)
                    x_j = tf.expand_dims(df_v[:, j], 1)

                    xij = tf.multiply(x_i, x_j)
                    if ffm_result is None:
                        ## 维度[batch, 1]
                        ffm_result = tf.reduce_sum(tf.multiply(wij, xij), axis=1, keepdims=True)
                    else:
                        ffm_result += tf.reduce_sum(tf.multiply(wij, xij), axis=1, keepdims=True)
            ffm_result = tf.reduce_sum(ffm_result, axis=1, keepdims=True)
        return ffm_result