import pandas as pd
import numpy as np
from math import log

class Entropy:
    '''
    计算离散随机变量的熵
    支持numpy和series数据类型
    '''
    @staticmethod
    def entropy(x):
        '''
        信息熵
        H(X)=-\sum p_i log2(p_i)
        '''
        x = pd.Series(x)
        p = x.value_counts(normalize=True)
        p = p[p > 0]
        h = -(p * np.log2(p)).sum()
        return h

    @staticmethod
    def cond_entropy(x, y):
        '''
        条件熵
        y必须是因子型/category变量
        H(X,y)=\sum p(y_i)H(X|y=y_i)
        '''
        y = pd.Series(y)
        x = pd.Series(x)
        p = y.value_counts(normalize=True)
        h = 0
        for yi in y.dropna().unique():
            h += p[yi] * Entropy.entropy(x[y == yi])
        return h
    
    @staticmethod
    def info_gain(x, y):
        '''
        信息增益==互信息
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        '''
        h = Entropy.entropy(x) - Entropy.cond_entropy(x, y)
        return h
    
class Dtree:
    '''
    构建决策树
    '''
    @staticmethod
    def max_info_gain_feature(data):
        '''
        选择信息增益最大的特征
        input:DataFrame格式数据集
        return:信息增益最大的特征编号
        '''
        dataset = np.array(data).tolist()
        labels = data.columns.to_list()
        feature_num = len(labels) - 1
        max_info_gain = 0.0
        max_info_feature = -1
        for i in range(feature_num):
            info_gain = Entropy.info_gain(data[labels[-1]], data[labels[i]])
            print("第%d个特征 %s 的信息增益为：%.3f" % (i, labels[i], info_gain))
            if (info_gain > max_info_gain):
                max_info_gain = info_gain
                max_info_feature = i
        return max_info_feature

    @staticmethod
    def create_dtree(data):
        '''
        使用ID3算法构建决策树
        input:DataFrame格式数据集
        return:决策树
        '''
        dataset = np.array(data).tolist()
        labels = data.columns.to_list()[:-1]
        class_list = [example[-1] for example in dataset]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(dataset[0]) == 1:
            return df.iloc[:, -1].value_counts().sort_values(
                ascending=False).index[0]

        #为了演示，临时使用全局变量j
        global j
        print(u"第%d：轮迭代" % (j))
        j = j + 1

        best_feature = Dtree.max_info_gain_feature(data)
        best_feature_name = labels[best_feature]
        print("本轮最优划分特征为：" + (best_feature_name) + "\n")
        dtree = {best_feature_name: {}}
        del (labels[best_feature])

        feature_list = data[best_feature_name].value_counts().index
        for value in feature_list:
            sub_data = data.loc[data[best_feature_name] == value].drop(
                [best_feature_name], axis=1)
            dtree[best_feature_name][value] = Dtree.create_dtree(sub_data)
        return dtree    
        