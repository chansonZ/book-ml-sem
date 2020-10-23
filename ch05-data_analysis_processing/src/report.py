# -*- coding: utf-8 -*-
# Chanson 21:30

import sys, os
import pandas as pd
import numpy as np

from dfutils import *


class DataMetaInfo():
    '''
    not support date type
    '''
    def __init__(self, df, copy=False, factor_threshold=10):
        '''#nas : list，用户定义na
        '''
        assert isinstance(df, pd.DataFrame)
        self.df = df.copy() if copy else df
        self.n_rows, self.n_cols = self.df.shape

        # 是否是数值
        self.num_idx = DFutils.is_numeric(self.df)
        self.num_cols = DFutils.ser_index(self.num_idx)

        # 是否是字符
        self.char_idx = (self.df.dtypes == object)
        self.char_cols = DFutils.ser_index(self.char_idx)

        # 统计指标
        # 空值列数、空值行数、唯一值数量
        print('na col row and unique stat')
        self.na_col_count = self.nan_stat(0)
        self.na_row_count = self.nan_stat(1)
        self.unique_count = DFutils.count_unique(self.df)

        # 是否是因子型，参考R
        print('factor stat')
        self.factor_idx = self.is_factor(factor_threshold)
        self.factor_cols = DFutils.ser_index(self.factor_idx)

        # 常量列，重复列
        print('constant_cols stat')
        self.constant_cols = self.stat_constant()

        # 全na统计
        print('all na stat')
        self.all_na_cols_idx = DFutils.all_na_cols(self.df, index=True)
        self.all_na_cols = DFutils.all_na_cols(self.df, index=False)
        self.all_na_rows = DFutils.all_na_rows(self.df)

        # 方差近0列
        print('near zero var stat')
        self.near_zero_idx = DataMetaInfo.near_zero_var(self.df)
        self.near_zero_var_cols = DFutils.ser_index(self.near_zero_idx)

        self.dup_cols = []
        self.dup_rows = []

        # 字符串长度统计
        print('max len stat')
        self.str_maxlen = DFutils.max_strlen_invalue(self.df)

        # 数据类型
        self.dtypes = self.df.dtypes.apply(lambda x: x.name)
        self.types = self.data_types()
        # 汇总信息
        # self._structure = pd.DataFrame()
        self.meta_info = None

        # 这个最好在接口中做为参数提供，因为大部分情况下用户并不知道数据情况
        self._nas = {'unknown', 'na', 'missing', 'n/a', 'not available'}

    def cal_duplicated(self):
        '''
        由于计算量很大，单独拿出来，当资源有限无法跑过时，再下一次报告时再运行
        '''
        print('duplicated cols and row stat')
        self.dup_cols = self.duplicated_cols()
        self.dup_rows = self.duplicated_rows()

    @staticmethod
    def sign_summary(df):
        '''每列正负数量统计
        more than describe
        建议/要求传入的df中都为数值型
        '''
        assert isinstance(df, pd.DataFrame), 'Input data is not pd.dataframe'
        s = pd.DataFrame(columns=[
            'NumOfNegative', 'PctOfNegative', 'NumOfPositive', 'PctOfPositive'
        ])

        s['NumOfPositive'] = df.apply(lambda x: (x > 0).sum(), axis=0)
        s['NumOfNegative'] = df.apply(lambda x: (x < 0).sum(), axis=0)
        s['NumOfZero'] = df.apply(lambda x: (x == 0).sum(), axis=0)
        s['PctOfPositive'] = s['NumOfPositive'] / df.shape[0]
        s['PctOfNegative'] = s['NumOfNegative'] / df.shape[0]
        s['PctOfZero'] = s['NumOfZero'] / df.shape[0]
        return s

    def nan_stat(self, axis=0):
        '''行,列的 Na 统计'''
        if axis == 0:
            t = DFutils.count_na_col(self.df)
            t = pd.DataFrame(t, columns=['NumOfNan'])
            t['PctOFNan'] = t['NumOfNan'] / self.n_rows
        elif axis == 1:
            t = DFutils.count_na_row(self.df)
            t = pd.DataFrame(t, columns=['NumOfNan'])
            t['PctOFNan'] = t['NumOfNan'] / self.n_cols
        return t

    def str_value_maxlen(self):
        ''' string 列最长值 '''
        return DFutils.max_len_in_strvalue(self.df)  #, self.char_cols)

    def stat_constant(self):
        col_to_keep = DFutils.sample_df(self.df).apply(
            lambda x: len(x.unique()) == 1, axis=0)
        if len(DFutils.ser_index(col_to_keep)) == 0:
            return []
        return DFutils.ser_index(self.df.loc[:, col_to_keep].apply(
            lambda x: len(x.unique()) == 1, axis=0))

    def is_factor(self, threshold=10):
        '''唯一值较少的统计'''
        threshold = threshold * self.n_rows if 0 < threshold < 1 else np.abs(
            threshold)
        return self.unique_count <= threshold

    def duplicated_cols(self, threshold=0.1):
        '''由于计算量大，做些优化—排除部分列
        '''
        cal_cols = [
            cc for cc in self.df.columns.tolist()
            if (cc not in self.near_zero_var_cols + self.all_na_cols)
        ]
        if len(cal_cols) == 0:
            print('No columns to cal duplicated')
            return []
        print(
            'There are {} cols to cal duplicated after remove near_zero_var_cols+all_na_cols ({})'
            .format(len(cal_cols),
                    len(set(self.near_zero_var_cols + self.all_na_cols))))

        # test,如果100行里没有重复的那必然就没有重复的
        df = self.df[cal_cols]
        if threshold < 1:
            threshold = int(threshold * df.shape[0])

        t = DFutils.sample_df(df, nr=threshold).T
        idx = (t.duplicated()) | (t.duplicated(keep='last'))
        if len(DFutils.ser_index(idx)) == 0:
            return []

        t = (df.loc[:, DFutils.ser_index(idx)]).T
        dup_index = t.duplicated()
        dup_index_complet = DFutils.ser_index((dup_index)
                                              | (t.duplicated(keep='last')))

        ll = []
        to_check_list = DFutils.ser_index(dup_index)
        check_cols = dup_index_complet
        while len(to_check_list) > 0 and len(check_cols) > 0:
            col = to_check_list.pop()
            index_temp = df[check_cols].apply(
                lambda x: (x == df[col])).sum() == self.n_rows

            # temp:即一组重复的列名。包括col本身，所以如果有重复的，只要在一组里随便保留一个即可
            temp = list(df[check_cols].columns[index_temp])
            if len(temp) > 0:
                ll.append(temp)
                for cc in temp:
                    if cc in to_check_list:
                        to_check_list.remove(cc)
                    if cc in check_cols:
                        check_cols.remove(cc)
        return ll

    def duplicated_rows(self, subset=None, return_df=False):
        if sum(self.df.duplicated()) == 0:
            print("there is no duplicated rows")
            return None
        if subset is not None:
            dup_index = (self.df.duplicated(subset=subset)) | (
                self.df.duplicated(subset=subset, keep='last'))
        else:
            dup_index = (self.df.duplicated()) | (self.df.duplicated(
                keep='last'))
        return dup_index

    @staticmethod
    def near_zero_var(df, freq_cut=95.0 / 5, unique_cut=10):
        nb_unique_values = DFutils.count_unique(df)
        n_rows, _ = df.shape
        percent_unique = 100 * nb_unique_values / n_rows

        def helper_freq(x):
            if nb_unique_values[x.name] == 0:
                return 0.0
            elif nb_unique_values[x.name] == 1:
                return 1.0
            else:
                t = x.value_counts()
                return float(t.iloc[0]) / t.iloc[1:].sum()

        freq_ratio = df.apply(helper_freq)

        zerovar = (nb_unique_values == 0) | (nb_unique_values == 1)
        near_zero = ((freq_ratio >= freq_cut) &
                     (percent_unique <= unique_cut)) | (zerovar)
        return near_zero

    def report_zero_var(self, top=2):
        ''' 必然有2个Top值,多个Top的话，输出格式就不固定了'''
        cols = self.near_zero_var_cols
        if len(cols) == 0:
            print('There is no zero_var columns~')
        ser_list = []
        for cc in cols:
            values = self.df[cc].value_counts().values[:top]
            v_list = [cc]
            na_pct = self.df[cc].isnull().sum() * 1.0 / self.df.shape[0]
            v_list.append(na_pct)

            sum_na = sum(self.df[cc].notnull())
            for tt in values:
                top_pct = tt * 1.0 / sum_na
                v_list.append(top_pct)
            if len(values) < top:
                t = len(values)
                while t < top:
                    v_list.append(np.nan)
                    t += 1
            ser_list.append(v_list)
        cols_name = ['Column', 'NaPct']
        cols_name += ['Top' + str(ii + 1) + 'Pct'
                      for ii in range(top)]  #python3
        return pd.DataFrame(ser_list, columns=cols_name)

    @staticmethod
    def metrics_of_numeric(df):
        '''more than describe
        建议/要求传入的df中都为数值型
        '''
        assert isinstance(df, pd.DataFrame), 'Input data is not pd.dataframe'

        col_order = [
            'Min', 'Max', 'Mean', 'Pct1', 'Pct5', 'Pct25', 'Pct50', 'Pct75',
            'Pct95', 'Pct99', 'Std', 'Mad', 'Skewness', 'Kurtosis'
        ]
        try:
            quantile_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            dfq = df.quantile(quantile_list).T
            dfq.rename(columns=dict(zip(quantile_list, col_order[3:10])),
                       inplace=True)
            dfq['Min'] = df.min()
            dfq['Max'] = df.max()
            dfq['Mean'] = df.mean()
            dfq['Std'] = df.std()
            dfq['Mad'] = df.mad()
            dfq['Skewness'] = df.skew()
            dfq['Kurtosis'] = df.kurt()
        except MemoryError:
            print(
                'MemoryError!!! So, all the numeric stat is Nan,you can try next time'
            )
            func_list = [[np.nan] * df.shape[1]] * len(col_order)
            return pd.DataFrame(func_list, index=col_order).T
        return dfq

    def data_types(self):
        dtypes_r = self.df.apply(lambda x: "character")
        dtypes_r[self.num_idx] = 'numeric'
        dtypes_r[self.factor_idx] = 'factor'
        return dtypes_r

    @staticmethod
    def report_numeric(df):
        print('metrics of numeric stat')
        metrics = DataMetaInfo.metrics_of_numeric(df)
        s = DataMetaInfo.sign_summary(df)
        return pd.concat([metrics, s], axis=1)

    def report_cols(self):
        self.types.name = 'types'
        self.dtypes.name = 'dtypes'
        self.unique_count.name = 'NumOfUnique'
        self.all_na_cols_idx.name = 'IsAllNa'
        self.near_zero_idx.name = 'NearZeroVar'
        self.str_maxlen.name = 'MaxLenOfStrValue'

        return pd.concat([
            self.types, self.dtypes, self.na_col_count, self.unique_count,
            self.all_na_cols_idx, self.near_zero_idx, self.str_maxlen
        ],
                         axis=1)

    def psummary(self):
        print('Data shape:\n{}\n'.format(self.df.shape))
        print('Data mem size:\n{:.3f}M\n'.format(
            self.df.memory_usage(index=True).sum() / 1024.0 / 1024.0))

        print('Sum of duplicated rows:\n{}\n'.format(
            sum(self.dup_rows) if self.dup_rows is not None else 'None'))
        print('Duplicated columns:\n{}\n'.format(
            self.dup_cols if self.dup_cols is not None else 'None'))
        print('Nero zero var columns:{}\n{}\n'.format(
            len(self.near_zero_var_cols), self.near_zero_var_cols))
        print('All na columns:{}\n{}\n'.format(len(self.all_na_cols),
                                               self.all_na_cols))
        print('All Na rows:{}\n{}\n'.format(
            sum(self.all_na_rows == True),
            DFutils.ser_index(self.all_na_rows)))

    def run(self):
        return self.report_zero_var(), DataMetaInfo.report_numeric(
            self.df[self.num_cols]), self.report_cols()
