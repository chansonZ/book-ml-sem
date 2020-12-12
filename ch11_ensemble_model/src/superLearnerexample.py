# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第11章 集成模型

import copy
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class SuperLearnerExample:
    '''base_models:基模型列表'''
    def __init__(self,
                 X,
                 y,
                 base_models,
                 meta_model,
                 kfolds=3,
                 test_size=0.3,
                 random_state=None):
        self.kfolds = kfolds
        self.base_models_oof = base_models
        self.base_models_all = copy.deepcopy(base_models)
        self.meta_model = meta_model
        self.random_state = random_state
        self.X, self.X_val, self.y, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)

    def _fit_base_models_all(self):
        for m in self.base_models_all:
            m.fit(self.X, self.y)

    def _get_oof_preds(self):
        meta_X, meta_y = [], []
        # cv
        kfold = KFold(n_splits=self.kfolds,
                      shuffle=True,
                      random_state=self.random_state)

        for train_ix, test_ix in kfold.split(self.X):
            fold_yhats = []
            train_X, test_X = self.X[train_ix], self.X[test_ix]
            train_y, test_y = self.y[train_ix], self.y[test_ix]
            meta_y.extend(test_y)

            # 在当前的折上，训练和折外预测
            for m in self.base_models_oof:
                m.fit(train_X, train_y)
                yhat = m.predict(test_X)
                fold_yhats.append(yhat.reshape(len(yhat), 1))
            meta_X.append(hstack(fold_yhats))

        return vstack(meta_X), asarray(meta_y)

    def fit(self):
        self._fit_base_models_all()
        meta_X, meta_y = self._get_oof_preds()
        self.meta_model.fit(meta_X, meta_y)

    def predict(self, X):
        '''预测过程
        1、在待预测集上使用基模型预测得到元模型的输入：meta_X
        2、meta_X输入到元模型得到集成模型的最终预测
        '''
        meta_X = []
        for m in self.base_models_all:
            yhat = m.predict(X)
            meta_X.append(yhat.reshape(len(yhat), 1))

        return self.meta_model.predict(hstack(meta_X))

    def evaluate_meta_models(self):
        preds = self.predict(self.X_val)
        print('Super Learner: MSE {:.3f}'.format(
            mean_squared_error(self.y_val, preds)))

            