#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 下午2:31
# @Author  : Zessay
"""
这个文件中的方法用于对数据进行处理
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

class FieldHandler(object):
    def __init__(self, train_file_path, test_file_path=None, category_columns=[], continuation_columns=[]):
        '''
        :param train_file_path: 训练集文件文件名
        :param test_file_path: 测试集文件文件名
        :param category_columns: 类别型特征, list型
        :param continuation_columns: 连续型特征, list型
        '''
        self.train_file_path = None
        self.test_file_path = None
        self.feature_nums = 0
        self.field_dict = {}


        self.category_columns = category_columns
        self.continuation_columns = continuation_columns

        if not isinstance(train_file_path, str):
            raise ValueError("train file path must str")
        if os.path.exists(train_file_path):
            self.train_file_path = train_file_path
        else:
            raise OSError("train file path isn't exists!")

        if test_file_path:
            if os.path.exists(test_file_path):
                self.test_file_path = test_file_path
            else:
                raise OSError("test file path isn't exists!")
        ## 读取数据
        self.read_data()
        ## 构建场到特征的字典
        self.build_field_dict()
        self.build_standard_scaler()
        self.field_nums = len(self.category_columns + self.continuation_columns)

    def read_data(self):
        '''
        读取数据
        '''
        if self.train_file_path and self.test_file_path:
            train_df = pd.read_csv(self.train_file_path)[self.category_columns + self.continuation_columns]
            test_df = pd.read_csv(self.test_file_path)[self.category_columns + self.continuation_columns]
            self.df = pd.concat([train_df, test_df])
        else:
            self.df = pd.read_csv(self.train_file_path)[self.category_columns + self.continuation_columns]

        self.df[self.category_columns] = self.df[self.category_columns].astype(str)


    def build_field_dict(self):
        '''
        构建场到特征的映射关系
        '''
        for column in self.df.columns:
            if column in self.category_columns:
                ## 类别型特征中所有不同的值
                '''
                ## 去掉缺失值
                cv = [f for f in self.df[column].unique() if str(f) != "nan"]
                '''
                ## 不去掉缺失值，将缺失值看做一种feature
                cv = [f for f in self.df[column].unique()]
                ## 将每一种特征值对应到不同的特征标号，键表示对应的场
                self.field_dict[column] = dict(zip(cv, range(self.feature_nums, self.feature_nums+len(cv))))
                ## 对应特征数增加
                self.feature_nums += len(cv)
            else:
                ## 对于连续型特征
                self.field_dict[column] = self.feature_nums
                self.feature_nums += 1

    def build_standard_scaler(self):
        '''
        对连续型特征进行标准化
        '''
        if self.continuation_columns:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(self.df[self.continuation_columns].values)
        else:
            self.standard_scaler = None



def transformation_data(df, field_handler, label=None):
    '''
    返回准备好的数据
    :param label: 目标值对应的列名
    :return:
    '''
    df_v = df.copy()
    if label:
        if label in df_v.columns:
            ## 获取对应的目标值
            labels = df_v[[label]].values.astype("float32")
        else:
            raise KeyError(f"label '{label}' isn\'t exists!")
    df_v = df_v[field_handler.category_columns+field_handler.continuation_columns]
    ## 对连续型特征和类别型特征的缺失值进行填充
    ##df_v[field_handler.category_columns].fillna("-1", inplace=True)
    ##df_v[field_handler.continuation_columns].fillna(-999, inplace=True)
    ## 对连续型特征进行归一化
    if field_handler.standard_scaler:
        df_v[field_handler.continuation_columns] = field_handler.standard_scaler.transform(df_v[field_handler.continuation_columns].values)

    ## 这个DataFrame用于记录每个类别型特征值和连续型特征对应的特征标号
    df_i = df_v.copy()

    for column in df_v.columns:
        if column in field_handler.category_columns:
            print("cat: ", column)
            df_i[column] = df_i[column].map(field_handler.field_dict[column])
            ## 对于测试集，可能有的特征值没有在训练集中出现
            '''
            ## 第0号特征留给缺失值
            df_i[column].fillna(0, inplace=True)
            ## 对非缺失值赋值为1
            df_v.loc[df_v[column].apply(lambda x: str(x) != "nan").values, column] = 1
            ## 对值序列的缺失值用0填充
            df_v[column].fillna(0, inplace=True)
            '''
            df_v[column] = 1 
        else:
            print("con: ", column)
            '''
            df_i.loc[np.isnan(df_v[column].values).astype("int") != 1, column] = field_handler.field_dict[column]
            df_i[column].fillna(0, inplace=True)
            '''
            df_i[column] = field_handler.field_dict[column]
            ## 对值列的缺失值用0填充
            df_v[column].fillna(0, inplace=True)

    df_v = df_v.values.astype("float32")
    df_i = df_i.values.astype("int32")
    features = {
        "df_i": df_i,
        "df_v": df_v
    }

    if label:
        return features, labels
    return features, None



def create_dirs(dirs):
    try:
        for dir_ in dirs: 
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0 
    except Exception as e:
        print("Creating directories error: {}".format(e))
        exit(-1)