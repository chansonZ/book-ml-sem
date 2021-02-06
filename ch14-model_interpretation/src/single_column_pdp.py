# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第14章-模型解释

import numpy as np
import pandas as pd

def generate_grid(df, features, resolution=100, grid_range=(.05, .95)):
    '''
    df：DataFrame
    features：待计算分位点的特征列名；
    该函数支持多维特征的特征值的排列组合，但PDP最多只支持2维特征
    '''
    from itertools import product
    grid_range = [x * 100 for x in grid_range]
    bins = np.linspace(*grid_range, num=resolution).tolist()
    grid = []
    for ff in features:
        data = df[ff]
        if len(np.unique(data)) < resolution:
            vals = np.unique(data)
        else:
            vals = np.unique(np.percentile(data, bins))
        grid.append(vals)
    grid = np.array(grid)
    # 返回笛卡尔积
    return pd.DataFrame(list(product(*grid))).values
    
def plot_single_column_pdp(model, df, column, target_label=1):
    ''' 适用于二分类模型，单变量pdp

    df : X，包含待分析的列column
    column: 待分析的单列名
    target_label:二分类时取预测列1的值
    '''
    assert isinstance(column, str), 'Need str column,only one column allowed!'
    x_cols = [column]
    grid_expanded = generate_grid(df, x_cols)

    indexs = [i for i in range(grid_expanded.shape[0])]
    pd_list = []
    for index in indexs:
        new_row = grid_expanded[index]
        pd_dict = {column: new_row[idx] for idx, column in enumerate(x_cols)}
        for feature_idx, feature_id in enumerate(x_cols):
            df_new = df.copy()

            df_new[feature_id] = new_row[feature_idx]
            try:
                probs = model.predict_proba(df_new)[:, target_label]
            except:
                probs = model.predict(df_new)

            mean_probs = np.mean(probs, axis=0)
            std_probs = np.std(probs, axis=0)  # 暂未使用

            pd_dict[target_label] = mean_probs
            pd_dict['std'] = std_probs
        pd_list.append(pd_dict)
    pd_list_df = pd.DataFrame(list(pd_list))
    pd_list_df.plot(x=column, y=target_label)
    
    