import numpy as np 
import pandas as pd 
import os 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.autonotebook import tqdm 

'''
对DataFrame内容进行处理，用于转换为FMFormat格式
'''

class FMFormat:
    def __init__(self, vector_feat, onehot_feat, continous_feat):
        self.feature_index = None  # 记录特征索引
        self.vector_feat = vector_feat
        self.onehot_feat = onehot_feat
        self.continous_feat = continous_feat
        
    def fit(self, df):
        self.feature_index = {}
        last_idx = 0
        for col in df.columns:
            ## 如果是one-hot型特征
            if col in self.onehot_feat:
                print("cat", col)
                df[col] = df[col].astype(str)
                ## 该变量对应多少种不同的值
                vals = [v for v in np.unique(df[col].values) if str(v) != "nan"]
                ## 获得对应的特征名
                names = np.asarray(list(map(lambda x: col+"_"+x, vals)))
                tmp = dict(zip(names, range(last_idx, last_idx+len(names))))
                self.feature_index.update(tmp)
                last_idx += len(names)
            elif col in self.vector_feat:
                ## 对于字符串类型的特征
                vals = []
                for data in df[col].astype(str).values:
                    if data != "nan":
                        ## 按照空格划分
                        for word in data.strip().split():
                            vals.append(word)
                vals = np.unique(vals)
                vals = filter(lambda x: x!="nan", vals)
                names = np.asarray(list(map(lambda x: col+"_"+x, vals)))
                tmp = dict(zip(names, range(last_idx, last_idx+len(names))))
                self.feature_index.update(tmp)
                last_idx += len(names)
            elif col in self.continous_feat:
                ## 如果是数值型特征
                print("con: ", col)
                self.feature_index.update({col:last_idx})
                last_idx += 1 
        return self 
    
    ## 对每一行进行转换
    def transform_row_(self, row):
        fm = []
        
        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.onehot_feat:
                if str(val) != "nan":
                    name = f"{col}_{val}"
                    if name in self.feature_index:
                        fm.append("{}:1".format(self.feature_index[name]))
            elif col in self.vector_feat:
                if str(val) != "nan":
                    for word in str(val).split():
                        name = f"{col}_{word}"
                        if name in self.feature_index:
                            fm.append("{}:1".format(self.feature_index[name]))
            elif col in self.continous_feat:
                if str(val) != "nan":
                    fm.append("{}:{}".format(self.feature_index[col], val))
        return " ".join(fm)
    
    def transform(self, df):
        return pd.Series({idx:self.transform_row_(row) for idx, row in df.iterrows()})
    
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


'''
转换为FM格式的接口函数
'''
def convert_to_fm(train_df, test_df=None, vector_fe=[], onehot_fe=[], contin_fe=[], path="./", label=None):
    train_ = train_df.copy()
    test_ = test_df.copy()
    
    if test_df is not None:
        df_ = pd.concat([train_, test_], axis=0, sort=False, ignore_index=True)
    else:
        df_ = train_
        
    trans = FMFormat(vector_fe, onehot_fe, contin_fe)
    user_fm = trans.fit_transform(df_)
    
    train_ = user_fm[:train_df.shape[0]]
    if test_df is not None:
        test_fm = user_fm[train_df.shape[0]:]
    
    if label:
        Y = train_df[label].values
    else:
        raise ValueError("Please give the label")
        
    train_fm = pd.DataFrame()
    train_fm['Label'] = Y.astype(str)
    train_fm['feature'] = train_
    train_fm['all'] = train_fm[['Label', "feature"]].apply(lambda row: " ".join(row),
                                                          axis=1, raw=True)
    train_fm.drop(["Label", "feature"], axis=1, inplace=True)
    
    ## 生成训练集和验证集
    ### 生成训练集
    train_string = ""
    for i in range(int(train_fm.shape[0]*0.8)):
        train_string += train_fm['all'].values[i]
        train_string += "\n"
    train_string = train_string.strip()
    with open(os.path.join(path, "train_fm.txt"), "w", encoding="utf8") as f: 
        f.write(train_string)
    
    ### 生成验证集
    valid_string = ""
    for i in range(int(train_fm.shape[0]*0.8), train_fm.shape[0]):
        valid_string += train_fm['all'].values[i]
        valid_string += '\n'
    valid_string = valid_string.strip()
    with open(os.path.join(path, "valid_fm.txt"), "w", encoding="utf8") as f: 
        f.write(valid_string)
    
    if test_df is not None:
        test_string = ""
        for i in range(test_fm.shape[0]):
            test_string += test_fm.values[i]
            test_string += "\n"
        test_string = test_string.strip()
        with open(os.path.join(path, "test_fm.txt"), "w", encoding="utf8") as f: 
            f.write(test_string)


'''
为把数据转换成FFM格式的训练类
'''
class FFMFormat:
    def __init__(self, vector_feat, one_hot_feat, continus_feat):
        '''
        vector_feat: 表示多个有意义的字符组成的特征，可以理解为向量型特征，缺失值用"-1"填充 
        one_hot_feat: 表示可以使用One-hot编码的特征，缺失值使用-1填充 
        continus_feat: 表示连续型特征，经过归一化处理的 
        '''
        self.field_index_ = None  # 记录场索引信息
        self.feature_index_ = None # 记录特征索引信息
        self.vector_feat = vector_feat
        self.one_hot_feat = one_hot_feat
        self.continus_feat = continus_feat
        
    def fit(self, df):
        ## 每一列对应一个场
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = {}
        last_idx = 0 
        for col in tqdm(df.columns):
            ## 如果对应列是one-hot型特征
            if col in self.one_hot_feat:
                print("cat: ", col)
                df[col] = df[col].astype(str)
                ## 求出该变量中共有多少种不同的值
                vals = [v for v in np.unique(df[col].values) if str(v) != "nan"]
                ## 获得对应的one-hot只有的特征名
                names = np.asarray(list(map(lambda x: col+"_"+x, vals)))
                tmp = dict(zip(names, range(last_idx, last_idx+len(names))))
                self.feature_index_[col] = tmp
                last_idx += len(names)
            elif col in self.vector_feat:
                ## 这是字符串型特征
                vals = []
                for data in df[col].apply(str):
                    if data != "nan":
                        ## 按照空格进行分割
                        for word in data.strip().split():
                            vals.append(word)
                vals = np.unique(vals)
                vals = filter(lambda x: x!="nan", vals)
                names = np.asarray(list(map(lambda x: col+"_"+x, vals)))
                tmp = dict(zip(names, range(last_idx, last_idx+len(names))))
                self.feature_index_[col] = tmp
                last_idx += len(names)
            elif col in self.continus_feat:
                ## 最后如果是数值型特征
                print("con: ", col)
                self.feature_index_[col] = last_idx
                last_idx += 1 
        return self 
    
    # 对每一行进行转换
    def transform_row_(self, row):
        ffm = []
        
        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = f"{col}_{val}"
                if name in self.feature_index_[col]:
                    ffm.append("{}:{}:1".format(self.field_index_[col], self.feature_index_[col][name]))
            elif col in self.vector_feat:
                for word in str(val).split():
                    name = f"{col}_{word}"
                    if name in self.feature_index_[col]:
                        ffm.append("{}:{}:1".format(self.field_index_[col], self.feature_index_[col][name]))
            elif col in self.continus_feat:
                if str(val) != "nan": 
                    ffm.append("{}:{}:{}".format(self.field_index_[col], self.feature_index_[col], val))
        return " ".join(ffm)
    
    def transform(self, df):
        return pd.Series({idx: self.transform_row_(row) for idx, row in tqdm(df.iterrows())})
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

'''
转换成FFM格式文件的接口类
'''
def convert_to_ffm(train_df, test_df=None, vector_fe=[], onehot_fe=[], contin_fe=[], path="./", label=None):
    
    train_ = train_df.copy()
    test_ = test_df.copy()
    
    if test_df is not None:
        df_ = pd.concat([train_, test_], axis=0, sort=False, ignore_index=True)
    else:
        df_ = train_
    
    trans = FFMFormat(vector_fe, onehot_fe, contin_fe)
    user_ffm = trans.fit_transform(df_)
    
    train_ = user_ffm[:train_df.shape[0]]
    if test_df is not None:
        test_ffm = user_ffm[train_df.shape[0]:]
    
    if label:
        Y = train_df[label].values
    else:
        raise ValueError("Please give the label")
    
    train_ffm = pd.DataFrame()
    train_ffm["Label"] = Y.astype(str) 
    train_ffm["feature"] = train_
    train_ffm['all'] = train_ffm[['Label', "feature"]].apply(lambda row: " ".join(row), axis=1, raw=True)
    train_ffm.drop(["Label", "feature"], axis=1, inplace=True)
    
    
    ## 生成训练集和验证集
    ### 生成训练集
    train_string = ""
    for i in range(int(train_ffm.shape[0]*0.8)):
        train_string += train_ffm['all'].values[i]
        train_string += "\n"
    train_string = train_string.strip()
    with open(os.path.join(path, "train_ffm.txt"), "w", encoding="utf8") as f: 
        f.write(train_string)
    
    ### 生成验证集
    valid_string = ""
    for i in range(int(train_ffm.shape[0]*0.8), train_ffm.shape[0]):
        valid_string += train_ffm['all'].values[i]
        valid_string += '\n'
    valid_string = valid_string.strip()
    with open(os.path.join(path, "valid_ffm.txt"), "w", encoding="utf8") as f: 
        f.write(valid_string)
    
    if test_df is not None:
        test_string = ""
        for i in range(test_ffm.shape[0]):
            test_string += test_ffm.values[i]
            test_string += "\n"
        test_string = test_string.strip()
        with open(os.path.join(path, "test_ffm.txt"), "w", encoding="utf8") as f: 
            f.write(test_string)


def preprocess(train_df, test_df=None, contin_fe=[], norm="ss"):
    '''
    对连续型特征进行归一化
    norm：表示使用的归一化方式，ss表示使用正态归一化，mm表示使用最大最小值归一化
    '''
    if test_df is not None:
        df_ = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    else:
        df_ = train_df
    
    if norm == "mm":
        trans = MinMaxScaler()
    else:
        trans = StandardScaler()
    df_[contin_fe] = trans.fit_transform(df_[contin_fe])
    
    train_df = df_[:train_df.shape[0]]
    if test_df is not None:
        test_df = df_[train_df.shape[0]:]
        return train_df, test_df
    return train_df, None

'''
对连续型特征进行分箱处理
'''
def cut_bins(train, test=None, contin_fe=[]):
    # 对跨度比较大的连续值进行分箱
    ## 分箱节点为：0, 25, 50, 75, 95, 100
    if test is not None:
        df_ = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    else:
        df_ = train
    ## 计算几个分位点
    ## 去除所有nan值
    for col in contin_fe:
        ### 去除nan值
        vals = df_[np.isnan(df_[col]).astype('int8') == 0][col].values
        Q0 = np.min(vals)
        Q1 = np.percentile(vals, 25)
        Q2 = np.percentile(vals, 50)
        Q3 = np.percentile(vals, 75)
        Q4 = np.percentile(vals, 95)
        Q5 = np.max(vals)
        bins = [Q0, Q1, Q2, Q3, Q4, Q5]
        bins = sorted(set(bins))
        labels = list(map(str, list(range(len(bins)-1))))
        df_[f"C_{col}"] = pd.cut(df_[col], bins=bins, labels=labels)
    
    train_ = df_[:train.shape[0]]
    if test is not None:
        test_ = df_[train.shape[0]:]
        return train_, test_
    return train_, None