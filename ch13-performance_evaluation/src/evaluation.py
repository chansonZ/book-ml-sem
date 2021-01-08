# -*- coding: utf-8 -*-
# 唐振
# 《机器学习：软件工程方法与实现》 第13章-模型评估

def show_prc(prob, actual, title="Model"):
    '''
    1. prob为模型预测的概率；即：prob = rf.predict_proba(X_test)
    2. actual为y_test,真实标签
    '''
    import sklearn.metrics
    import matplotlib.pyplot as plt
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(actual, prob, pos_label=1)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(actual, prob, pos_label=1)
    ret_auc = sklearn.metrics.auc(fpr, tpr, reorder=True)
    fig = plt.figure(figsize=(8,6)) #,dpi=300)
    plt.title('PRC Curve - ' + title, fontsize=15)
    plt.plot(precision, recall,'b',label='AUC:%0.2f'%ret_auc, linewidth=3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision',fontsize=15)
    plt.xlabel('Recall',fontsize=15)
    plt.show()


import sklearn.metrics
from sklearn.metrics import *
def show_roc(prob, actual, title="Model"):
    '''
    1. prob为模型预测的概率；即：prob = rf.predict_proba(X_test)
    2. actual为y_test,真实标签
    '''
    fpr, tpr, threshold = roc_curve(actual, prob)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    kss = sorted([(x - y, x, y) for x, y in zip(tpr, fpr)],
                 key=lambda x: x[0], reverse=True)
    max_ks = kss[0]
    ks_max = max_ks[0]
    tpr_max = max_ks[1]
    fpr_max = max_ks[2]
    # method I: plt
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic' + ' - ' + title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f, KS= %0.2f' %
             (roc_auc, ks_max), linewidth=4)
    plt.plot([fpr_max], [tpr_max], label='KS Point(TPR=%0.2f, FPR=%0.2f)'%
             (tpr_max, fpr_max), marker='o', markersize=16, color="green")
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def show_ks(prob, actual, title="Model"):
    '''
    说明：
    1. prob为模型预测的概率；即：prob = rf.predict_proba(X_test)
    2. actual为y_test,真实标签
    '''
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(actual, prob)
    ks_ary = list(map(lambda x, y: x - y, tpr, fpr))
    ks = np.max(ks_ary)
    y_axis = list(map(lambda x: x * 1.0 / len(fpr), range(0, len(fpr))))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    plt.title('K-S CURVE' + ' - ' + title)
    plt.plot(fpr, y_axis, 'b', linewidth=4, label='fpr')  # fpr曲线；bad的曲线
    plt.plot(tpr, y_axis, 'y', linewidth=4, label='tpr')     # TPR分对的曲线；
    plt.plot(y_axis, ks_ary, 'g', linewidth=4, label='KS= %0.2f' % (ks))       # ks曲线
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
def cal_vif(df, cols):
    """
    df: pandas中DataFrame
    cols: 建模特征列名称
    """
    data = df.assign(Intercept=1)[['Intercept'] + cols].values
    vif_list = [variance_inflation_factor(data, i) 
				for i in range(len(cols)+1)][1:]
    return pd.DataFrame(list(zip(cols, vif_list)), columns=['col', 'vif'])


