# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第5章 数据分析与处理

import sys,os
import pandas as pd
import numpy as np
from numpy.random import permutation

class DFutils():
    '''
    DataFrame 上操作
    '''
    @staticmethod
    def ser_index(series, index=False):
        '''返回为true的index(Index数组)或index的名字'''
        if index:
            return series[series].index
        else:
            return series[series].index.tolist()

    @staticmethod
    def is_df(data):
        return isinstance(data, pd.DataFrame) if data is not None else False

    @staticmethod
    def to_lowercase(df):
        ''' 返回转化小写后的整个新df'''
        return df.applymap(lambda x: x.lower() if type(x) == str else x)

    @staticmethod
    def where_numeric(df):
        ''' 返回是否是数值boolean的新df'''
        return df.applymap(lambda x: isinstance(x, (int, float)))

    @staticmethod
    def count_unique(df):
        ''' 计算每列唯一值数量，排除NA:nunique: Excludes NA '''
        return df.apply(lambda x: x.nunique(), axis=0)

    @staticmethod
    def count_na_col(df):
        '''每列 Na count'''
        return df.isnull().sum(axis=0)

    @staticmethod
    def count_na_row(df):
        '''每行 Na count'''
        return df.isnull().sum(axis=1)

    @staticmethod
    def sample_df(df, pct=0.1, nr=100):
        ''' 采样 随机取a行'''
        a = max(int(pct * df.shape[0]), int(nr))
        return df.loc[permutation(df.index)[:a], :]

    @staticmethod
    def specific_str_value(df, values):
        '''check user specified value
        df: string
        values:list of value of string
        '''
        values = values if isinstance(values, list) else [values]
        ret = DFutils.to_lowercase(df).applymap(lambda x: x in values)
        print("{} has {} count in dadaframe".format(values, ret.sum().sum()))
        return ret

    @staticmethod
    def fill_na_with_npna(df, str_nas):
        '''整个df中替换
        str_nas:用户指定的为na的字符串列表，df中是该列表的值时将替换为np.nan
        example:
                fill_na_with_npna(df,['na','n/a'])
        '''
        na_idx = DFutils.specific_str_value(df, str_nas)
        return df.where((na_idx == False), np.nan)

    @staticmethod
    def all_na_cols(df, index=False):
        '''全是缺失值的列
        '''
        if index:
            return DFutils.count_na_col(df) == df.shape[0]
        else:
            return DFutils.ser_index(DFutils.count_na_col(df) == df.shape[0])

    @staticmethod
    def all_na_rows(df):
        '''全是缺失值的行索引
        '''
        return DFutils.count_na_row(df) == df.shape[1]

    @staticmethod
    def is_numeric(df, colname=None):
        '''
        return 
            True or False
        '''
        if colname is None:
            colname = df.columns.tolist()
        dtype_col = df.loc[:, colname].dtypes
        t = (dtype_col == int).values | (dtype_col == float).values
        return pd.Series(t, index=dtype_col.index)

        #return (dtype_col == int) or (dtype_col == float)

    @staticmethod
    def is_numeric_bycount(df, count_thresh=10):
        '''
        ret = 
        return DFutils.ser_index( ret>=count_thresh)
        
        return
            column name
        '''
        return DFutils.ser_index(DFutils.count_unique(df) >= count_thresh)

    @staticmethod
    def max_strlen_invalue(df, str_cols=None):
        '''str_cols:list of str columns'''
        if str_cols is not None:
            return df[str_cols].apply(lambda x: np.max(x.str.len()), axis=0)
        else:
            return df.apply(
                lambda x: np.max(x.str.len()) if x.dtype.kind == 'O' else np.nan,
                axis=0)
