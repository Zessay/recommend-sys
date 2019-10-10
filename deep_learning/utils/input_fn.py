#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 下午3:32
# @Author  : Zessay

'''
定义训练时数据的生成过程
'''
import numpy as np
from collections import Counter

class DataGenerator:
    def __init__(self, labels, *features):
        self.features = features
        self.labels = labels
        self.length = len(labels)
        ## 计算不同类别的比例
        unique = Counter(self.labels.ravel())
        self.ratio = [(key, value / self.length) for key, value in unique.items()]
        self.indices = []
        for key, _ in self.ratio:
            index = np.where(labels.ravel() == key)
            self.indices.append(index)
        
    def next_batch(self, batch_size):
        '''
        生成每一个batch的数据集
        '''
        choose = np.array([])
        for i in range(len(self.indices)):
            ## 按照在数据集中出现的比例采样
            idx = np.random.choice(self.indices[i][0],
                                   max(1, min(len(self.indices[i][0]), int(batch_size * self.ratio[i][1]))))
            '''
            ## 等比例采样
            idx = np.random.choice(self.indices[i][0],
                                  min(len(self.indices[i][0]), int(batch_size / len(self.indices))))
            '''
            choose = np.append(choose, idx)
        choose = np.random.permutation(choose).astype("int64")
        result = []
        for feat in self.features:
            result.append(feat[choose])
        result.append(self.labels[choose])
        yield result
        
    def iter_all(self, batch_size):
        '''
        按照batch迭代所有数据
        '''
        numBatches = self.length // batch_size + 1 
        for i in range(numBatches):
            result = []
            start = i*batch_size
            end = min(start+batch_size, self.length)
            for feat in self.features:
                result.append(np.asarray(feat[start:end]))
            result.append(np.asarray(self.labels[start:end]))
            yield result