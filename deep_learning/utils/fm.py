#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 上午10:32
# @Author  : Zessay

import tensorflow as tf
import numpy as np

class FM(object):
    def __init__(self, hparams, df_i, df_v):
        '''
        :param hparams: 这是一个nametuple类型或者一个对象，包含需要指定的参数
        :param df_i: 这里是 [batch, n]大小的张量，表示输入特征的索引，n表示特征的维度
        :param df_v: 这里是 [batch, n]大小的张量，表示输入特征的值
        '''
        self.hparams = hparams
        tf.set_random_seed(self.hparams.seed)
        self.load_activation_fn()
        # 这里得到线性模型的结果
        self.line_result = self.line_section(df_i, df_v)
        # 这里得到二阶特征组合的结果
        self.fm_result = self.fm_section(df_i, df_v)

        # 如果不使用深度模型，则直接将两个值相加即可
        if not self.hparams.use_deep:
            self.logits = self.line_result + self.fm_result
        else:
            ## 获取深度网络的输入
            self.deep_result = self.deep_section()
            self.deep_result = tf.nn.dropout(self.deep_result, keep_prob=self.hparams.deep_output_keep_dropout)
            self.line_result = tf.nn.dropout(self.line_result, keep_prob=self.hparams.line_output_keep_dropout)
            self.fm_result = tf.nn.dropout(self.fm_result, keep_prob=self.hparams.fm_output_keep_dropout)
            concat = tf.concat(values=[self.line_result, self.fm_result, self.deep_result], axis=1)
            self.logits = tf.layers.dense(concat, units=1, activation=None)

    def line_section(self, df_i, df_v):
        with tf.variable_scope("line"):
            # 这里权重的维度为 features*1，表示一阶特征权重
            weights = tf.get_variable("weights", shape=[self.hparams.feature_nums, 1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.glorot_uniform())
            # 大小为 [batch, n, 1]
            batch_weights = tf.nn.embedding_lookup(weights, df_i)
            batch_weights = tf.squeeze(batch_weights, axis=2)
            # 得到特征线性组合的结果, [batch, n]
            line_result = tf.multiply(df_v, batch_weights, name="line_w_x")
            # 加上偏置
            bias = tf.get_variable("bias",
                                   shape=[1, 1],
                                   dtype=tf.float32,
                                   initializer=tf.initializers.zeros())
            # 维度 [batch, 1]
            line_result = tf.add(tf.reduce_sum(line_result, axis=1, keepdims=True), bias)
        return line_result

    def fm_section(self, df_i, df_v):
        with tf.variable_scope("fm"):
            # 这里得到隐向量的初始值，维度为 [features, embedding_size]
            embedding = tf.get_variable("embedding",
                                        shape=[self.hparams.feature_nums,
                                               self.hparams.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.random_normal())
            # 得到输入特征的隐向量
            batch_embedding = tf.nn.embedding_lookup(embedding, df_i)
            df_v = tf.expand_dims(df_v, axis=2)  # [batch, n, 1]

            # 将特征值和对应的隐向量相乘
            self.xv = tf.multiply(df_v, batch_embedding)  # [batch, n, embedding_size]
            # 得到维度 [batch, embedding_size]
            sum_square = tf.square(tf.reduce_sum(self.xv, axis=1))
            square_sum = tf.reduce_sum(tf.square(self.xv), axis=1)

            fm_result = 0.5 * (tf.subtract(sum_square, square_sum))
            if self.hparams.use_deep:
                return fm_result
            # 维度 [batch, 1]
            fm_result = tf.reduce_sum(fm_result, axis=1, keepdims=True)

        return fm_result

    def load_activation_fn(self):
        if self.hparams.activation == "relu":
            self.activation = tf.nn.relu
        elif self.hparams.activation == "tanh":
            self.activation = tf.nn.tanh
        elif self.hparams.activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        elif self.hparams.activation == "elu":
            self.activation = tf.nn.elu
        else:
            raise ValueError("Please input correct activation function!")

    def deep_section(self):
        # 将上面fm得到的结果转换成指定的形状，主要是field也要作为维度
        deep_input = tf.reshape(self.xv, [-1, self.hparams.field_nums*self.hparams.embedding_size], name="deep_input")
        deep_input = tf.nn.dropout(x=deep_input, keep_prob=self.hparams.deep_input_keep_dropout)
        for i, v in enumerate(self.hparams.layers):
            deep_input = tf.layers.dense(deep_input, units=v, activation=None)
            if self.hparams.use_batch_norm:
                deep_input = tf.layers.batch_normalization(deep_input)
            deep_input = self.activation(deep_input)
            if (i+1) != len(self.hparams.layers):
                deep_input = tf.nn.dropout(deep_input, self.hparams.deep_mid_keep_dropout)
        return deep_input
