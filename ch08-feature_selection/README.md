# 请参考书中内容实践

不急，慢点学习。

# StepWise 
前向选择、后向选择、前-后向、后-前算法实现，向请参考书中介绍和源码

- forward
- backward
- fb
- bf

# 特征选择算法包使用示例

## 导入样例数据

```python
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()

# 转化为df
X = pd.DataFrame.from_records(data=data.data, columns=data.feature_names)
df = X
df['target'] = data.target
df.shape

    (150, 5)
```

## 准备待筛选的特征列

```python
x_col = [cc for cc in df.columns if cc != 'target']
x_col

['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']
```

## 运行特征选择算法包


```python
sf = SelectFeatures(df[x_col], df['target'])#,n_features_to_select=3)

self.n_features_to_select: 3
```

```python
# 10个选择算法集成
selected, stat_f = sf.example_10_methods()


    ***by_var:
      remain feature index:
      [0 1 2 3]
      removed feature index:
      []
    
    ***by_pearson:
      remain feature index:
      [0 2 3]
      removed feature index:
      [1]
    
    ***by_max_info:
      remain feature index:
      [0 2 3]
      removed feature index:
      [1]
    
    ***by_REF_svm:
      remain feature index:
      [1 2 3]
      removed feature index:
      [0]
    
    ***by_REF_lr:
      remain feature index:
      [1 2 3]
      removed feature index:
      [0]
    
    ***by_svm:
      remain feature index:
      [0 1 2]
      removed feature index:
      [3]
    
    ***by_lr:
      remain feature index:
      [0 1 2]
      removed feature index:
      [3]
    
    ***by_et:
      remain feature index:
      [2 3]
      removed feature index:
      [0, 1]
    



    ***by_rf:
      remain feature index:
      [2 3]
      removed feature index:
      [0, 1]
    
    ***by_gbdt:
      remain feature index:
      [2 3]
      removed feature index:
      [0, 1]
    
    **********remains columns:
    ['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']
```

## 查看选择结果

```python
selected

# 输出选择的特征顺序

 ['petal length (cm)', 'petal width (cm)', 'sepal length (cm)']
```
   
```python
# 特征选择明细

'''
例如：
petal length (cm) 被10个选择算法选中，排名第一
sepal length (cm) 被5个算法选中：by_var,by_max_info,by_pearson,by_svm,by_lr
'''
stat_f
```


|  | col_idx | count | feature |            by_var | by_max_info | by_pearson | by_RFE_svm | by_RFE_lr | by_svm | by_lr | by_et | by_rf | by_gbdt |      |
| ------: | ------: | ----: | ------: | ----------------: | ----------: | ---------: | ---------: | --------: | -----: | ----: | ----: | ----: | ------: | ---- |
|       0 |     2 |      10 | petal length (cm) |           1 |          1 |          1 |         1 |      1 |     1 |     1 |     1 |       1 | 1    |
|       1 |     3 |       8 |  petal width (cm) |           1 |          1 |          1 |         1 |      1 |     0 |     0 |     1 |       1 | 1    |
|       3 |     0 |       5 | sepal length (cm) |           1 |          1 |          1 |         0 |      0 |     1 |     1 |     0 |       0 | 0    |
|       2 |     1 |       5 |  sepal width (cm) |           1 |          0 |          0 |         1 |      1 |     1 |     1 |     0 |       0 | 0    |
