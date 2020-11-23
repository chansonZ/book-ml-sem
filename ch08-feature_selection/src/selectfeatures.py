# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》第8章-特征选择代码

import numpy as np
import pandas as pd
from minepy import MINE
from scipy.stats import pearsonr

from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, f_regression

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def list_diff(list1, list2):
    """return: 两个list之间的差集"""
    if len(list1) > 0 and len(list2) > 0:
        return list(np.setdiff1d(list1, list2))
    else:
        print('list_diff:len <=0 !!')


class SelectFeatures():
    '''
    X:pandas.DataFrame
    y:pandas.serise 或 nparray
    n_features_to_select:选择特征的数
    only_get_index：是否只返回选中特征的索引
    '''
    def __init__(self, X, y, n_features_to_select=None, only_get_index=True):
        self.cols = X.columns.tolist()
        self.X = np.array(X)
        self.y = np.array(y)
        self.x_index = range(self.X.shape[1])
        self.only_get_index = only_get_index
        self.n_features_to_select = n_features_to_select
        if n_features_to_select is None:
            self.n_features_to_select = int(np.ceil(2 / 3 * self.X.shape[1]))
            print('self.n_features_to_select:', self.n_features_to_select)
        self.removed = []

    def _log(self, index, method):
        print('***{}:'.format(method))
        print('  remain feature index:\n  {}'.format(index))
        rmvd = list_diff(self.x_index, index)
        self.removed += rmvd
        print('  removed feature index:\n  {}\n'.format(rmvd))

    def _return(self, ret, method):
        # True代表该特征被选中
        index = ret.get_support(indices=True)
        self._log(index, method)

        if self.only_get_index == True:
            return index
        else:  #返回筛选之后的X
            return ret.transform(self.X)

    # Filter方法
    def _by_kbest(self, func, method):
        ret = SelectKBest(func,
                          k=self.n_features_to_select).fit(self.X, self.y)
        return self._return(ret, method)

    # Wrapper方法
    def _by_RFE(self, mm, method, step=1):
        ret = RFE(estimator=mm,
                  n_features_to_select=self.n_features_to_select,
                  step=step).fit(self.X, self.y)
        return self._return(ret, method)

    # Embedded方法
    def _by_model(self, mm, method):
        ret = SelectFromModel(mm).fit(self.X, self.y)
        return self._return(ret, method)

    # stat
    def by_var(self, threshold=0.16):
        ret = VarianceThreshold(threshold=threshold).fit(self.X)
        return self._return(ret, 'by_var')

    def by_chi2(self):
        return self._by_kbest(chi2, 'by_chi2')

    def by_pearson(self):
        ''' 相关系数法 '''
        _pp = lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))
                                    ).T[0]
        return self._by_kbest(_pp, 'by_pearson')

    def by_max_info(self):
        # or mutual_info_classif
        def _mic(x, y):
            m = MINE()
            m.compute_score(x, y)
            return (m.mic(), 0.5)

        _pp = lambda X, Y: np.array(list(map(lambda x: _mic(x, Y), X.T))).T[0]
        return self._by_kbest(_pp, 'by_max_info')

    def by_f_regression(self):
        '''
        return:
            F values of features.
            p-values of F-scores.
        '''
        ret = f_regression(self.X, self.y)
        print('Feature importance by f_regression:{}'.format(ret))
        return ret

    def by_f_classif(self):
        ret = f_classif(self.X, self.y)
        print('Feature importance by f_regression:{}'.format(ret))
        return ret

    def by_RFE_lr(self, args=None):
        return self._by_RFE(LogisticRegression(), 'by_REF_lr')

    def by_RFE_svm(self, args=None):
        return self._by_RFE(LinearSVC(), 'by_REF_svm')

    def by_gbdt(self):
        return self._by_model(GradientBoostingClassifier(), 'by_gbdt')

    def by_rf(self):
        return self._by_model(RandomForestClassifier(), 'by_rf')

    def by_et(self):
        return self._by_model(ExtraTreesClassifier(), 'by_et')

    def by_lr(self, C=0.1):
        return self._by_model(LogisticRegression(penalty='l1', C=C,solver='liblinear'), 'by_lr')

    def by_svm(self, C=0.01):
        return self._by_model(LinearSVC(penalty='l1', C=C, dual=False),
                              'by_svm')

    # 演示示例
    def example_10_methods(self):
        name = [
            'by_var', 'by_max_info', 'by_pearson', 'by_RFE_svm', 'by_RFE_lr',
            'by_svm', 'by_lr', 'by_et', 'by_rf', 'by_gbdt'
        ]
        # {0:col_0,1:col_1}
        map_index_cols = dict(zip(range(len(self.cols)), self.cols))

        # 执行特征选择算法
        method_dict = {}
        method_dict['by_var'] = self.by_var()
        method_dict['by_pearson'] = self.by_pearson()
        method_dict['by_max_info'] = self.by_max_info()
        method_dict['by_RFE_svm'] = self.by_RFE_svm()
        method_dict['by_RFE_lr'] = self.by_RFE_lr()
        method_dict['by_svm'] = self.by_svm()
        method_dict['by_lr'] = self.by_lr()
        method_dict['by_et'] = self.by_et()
        method_dict['by_rf'] = self.by_rf()
        method_dict['by_gbdt'] = self.by_gbdt()

        # 打平选中特征的list
        selected = [j for i in list(method_dict.values()) for j in i]

        # 构建特征被哪些方法选中：0，1 表示
        dicts01 = {}
        for nm in name:
            dicts01[nm] = [
                1 if i in list(method_dict[nm]) else 0
                for i in range(len(self.cols))
            ]

        # 构建结果统计用的DataFrame
        stat_f = pd.Series(selected).value_counts().reset_index()
        stat_f.columns = ['col_idx', 'count']
        stat_f['feature'] = stat_f.col_idx.map(map_index_cols)

        # 升序排列匹配模型选择方法的值
        stat_f.sort_values(by='col_idx', ascending=True, inplace=True)

        for i in name:
            stat_f[i] = dicts01[i]

        # 按照特征被选中个数降序排列, 个数相同的情况下按照idx升序排列
        stat_f.sort_values(by=['count', 'col_idx'],
                           ascending=[False, True],
                           inplace=True)

        selected = stat_f['feature'][:self.n_features_to_select].tolist()
        print('*' * 10 + 'remains columns:\n{}'.format(selected))

        return selected, stat_f