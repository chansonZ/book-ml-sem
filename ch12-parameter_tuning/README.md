# 第12章  模型调参

相关数据和代码


# gridsearch_cv示例

```python

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 人工数据集
X, y = make_blobs(n_samples=1000,
                  centers=2,
                  n_features=10,
                  cluster_std=10,
                  random_state=42)

# 定义学习算法：使用逻辑回归
model = LogisticRegression()

# 定义超参空间
# 1、定义数值求解方法-类别型
solvers = ['newton-cg', 'lbfgs', 'liblinear']

# 2、定义正则项-类别型
penalty = ['l2']

# 3、定义惩罚力度-对数型
c_values = [100, 10, 1.0, 0.1, 0.01]

# 运行网格搜索
grid = dict(solver=solvers, penalty=penalty, C=c_values)

cv_best_model = classify_gridsearch_cv(model, X, y, grid)

```

输出结果

```python

Best: 0.779 : {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}

0.778 [0.034] : {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
0.778 [0.034] : {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
0.778 [0.034] : {'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}
0.778 [0.034] : {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
0.778 [0.034] : {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
0.778 [0.034] : {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
0.778 [0.034] : {'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'}
0.778 [0.034] : {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
0.778 [0.034] : {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}
0.778 [0.034] : {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
0.778 [0.034] : {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
0.779 [0.034] : {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
0.779 [0.034] : {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
0.779 [0.034] : {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
0.778 [0.033] : {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}

```

# XGBoost调参工具包使用示例

使用示例

1）准备数据：

```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
```

2)调参工具类初始化：

```
tb = TuneXGB(X_train,y_train)
```

3)查看缺省调参顺序：
```
tb.show_default_order()

# 输出
1 step:{'n_estimators': range(100, 1000, 50)}
 2 step:{'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1]}
 3 step:{'max_depth': [3, 5, 7, 9, 12, 15, 17, 25]}
 4 step:{'min_child_weight': [1, 3, 5, 7]}
 5 step:{'gamma': [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
 6 step:{'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
 7 step:{'colsample_bytree': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]}
 8 step:{'reg_alpha': [0, 0.1, 0.5, 1.0, 10, 100, 200, 1000]}
 9 step:{'reg_lambda': [0.01, 0.1, 1.0, 10, 100, 200, 1000]}

```

4)查看缺省参数：

```
tb.show_default_para()

# 输出

{'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 1}
```

5)调参过程中随时更新参数：
```
tb.update_cur_params({'n_estimators':180})
```

6)自定义调参：

```
tb.tune_step({'n_estimators': range(30, 100, 10)},verbose=3)
# 输出
Fitting 5 folds for each of 7 candidates, totalling 35 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
------------------------------------------------------------
Tunning:{'n_estimators': range(30, 100, 10)}
    use metric:roc_auc,folds:5
   Best: 0.98818 using {'n_estimators': 90}
mean,stdev,param:
   0.98705 (0.01222) with: {'n_estimators': 30}
……输出省略
   0.98818 (0.01103) with: {'n_estimators': 90}
Best params:
    {'n_estimators': 90}

Save param at:3
Save estimator at:3
[Parallel(n_jobs=4)]: Done  35 out of  35 | elapsed:    0.7s finished
```

7)增量调参：

```
tb.tune_sequence()
```

8)获取当前学习器：

```
tb.get_cur_estimator()
```

9)查看历史参数和学习器：
```
tb.history_paras
tb.history_estimator
```

10）进行网格搜索：

```
tb_gs = tb.grid_search({'reg_lambda': [0.01, 0.1, 1.0, 10, 100, 200, 1000]})
```

11)进行随机搜索：
```
tb_rs = tb.random_search({'colsample_bytree_loc':0.3})
```


