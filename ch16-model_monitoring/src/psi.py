# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第16章 模型稳定性监控

import numpy as np
import pandas as pd

def psi(expect,
        actual,
        bin_nums=10,
        return_dict=True,
        unique_threshold=10,
        dropna=True):
    """ Population Stability Index
    参考：《信用风险评分卡研究-12章》

    test
        rndm = np.random.RandomState(1234)
        e = rndm.normal(size=10**2)
        a = rndm.normal(size=10**2)
        psi_d = psi(pd.Series(e),pd.Series(a))
    """
    def _psi(a, e):
        if np.sum(a == 0) > 0:
            print('actual data contains zero !! return 1')
            return 1
        return np.sum((a - e) * np.log(a / e))

    def _fillna_cat(t, dropna):
        t0 = pd.isnull(t[0]).sum()
        t1 = pd.isnull(t[1]).sum()
        if t0 > 0 or t1 > 0:
            print('Nan statistics:\n{}'.format(t0 + t1))
            if dropna:
                print('    Drop Nan !!')
                t = t.dropna()
            else:
                print('Replace Nan with min/10.0 !!')
                t.fillna(tt.min().min() / 10.0, inplace=True)
        return t

    def _fillna_cont(t1, t2, dropna):
        # 注意，不同 pandas 版本差异isna，isnull
        tt1 = t1.isna().sum()
        tt2 = t2.isna().sum()

        if tt1 > 0 or tt2 > 0:
            print('Nan statistics for expect:\n{}'.format(tt1))
            print('Nan statistics for actual:\n{}'.format(tt2))
            if dropna:
                print('    Drop Nan !!')
                t1 = t1.dropna()
                t2 = t2.dropna()
            else:
                fillvalue = np.min(t1.min(), t2.min()) - 1
                t1.fillna(fillvalue, inplace=True)
                t2.fillna(fillvalue, inplace=True)
        return t1, t2

    def _bin_format(b):
        b = np.unique(b)
        b[0] = -np.inf
        b[-1] = +np.inf
        return b

    if len(np.unique(expect)) < unique_threshold:
        e_pct = expect.value_counts() / len(expect)
        a_pct = actual.value_counts() / len(actual)

        e_pct = e_pct.sort_index()
        a_pct = a_pct.sort_index()

        t = pd.concat([e_pct, a_pct], axis=1)
        t.columns = [0, 1]

        t = _fillna_cat(t, dropna)
        e_pct, a_pct = t[0], t[1]
    else:
        expect, actual = _fillna_cont(expect, actual, dropna)

        bins = np.percentile(expect, [(100.0 / bin_nums) * i
                                      for i in range(bin_nums + 1)],
                             interpolation="nearest")
        bins = _bin_format(bins)
        e_pct = (pd.cut(expect, bins=bins,
                        include_lowest=True).value_counts()) / len(expect)
        a_pct = (pd.cut(actual, bins=bins,
                        include_lowest=True).value_counts()) / len(actual)

        a_pct = a_pct.sort_index()
        e_pct = e_pct.sort_index()

    p = _psi(a_pct, e_pct)
    if return_dict:
        results = pd.DataFrame(
            {
                'expect_pct': e_pct.values,
                'actual_pct': a_pct.values
            },
            index=e_pct.index)
        return {'data': results, 'statistic': p}
    return p


# 进一步封装成可直接计算DataFrame的psi，请读者自行尝试
def cal_df_psi(df_exp, df_act, bin_nums=10):
    '''
    df_exp,df_act 要求是 dataFrame 格式,所以单列需要 train_x[['col']]
    '''
    assert isinstance(df_exp, pd.DataFrame), 'Need DataFrame'
    assert isinstance(df_act, pd.DataFrame), 'Need DataFrame'
    assert df_exp.shape[1] == df_act.shape[
        1], 'df_exp,df_act should be same shape[1]'

    cols = df_exp.columns.tolist()
    col_name = []
    psis = []
    for cc in cols:
        print('cal psi: {}'.format(cc))
        psis.append(
            psi(df_exp[cc], df_act[cc], bin_nums=bin_nums, return_dict=False))
        col_name.append(cc)
    return pd.DataFrame({'column': col_name, 'psi': psis})
    