# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第12章-模型调参

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

def classify_gridsearch_cv(model,
                           X,
                           y,
                           grid,
                           folds=10,
                           n_repeats=3,
                           scoring='accuracy',
                           seed=42):
    cv = RepeatedStratifiedKFold(n_splits=folds,
                                 n_repeats=n_repeats,
                                 random_state=seed)
    gs = GridSearchCV(estimator=model,
                      param_grid=grid,
                      n_jobs=-1,
                      cv=cv,
                      scoring=scoring,
                      error_score=0)
    gs = gs.fit(X, y)
    print("Best: {:.3f} : {}\n".format(gs.best_score_, gs.best_params_))
    
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    params = gs.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("{:.3f} [{:.3f}] : {}".format(mean, stdev, param))
        
    # 返回最好的模型
    return gs.best_estimator_

    