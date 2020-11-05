
之所以没有直接给jupyter notebook而是这里的文档,是希望大家实践时慢一点，思考一下，然后再执行。

# import

```python
import sklearn
import pandas as pd
import numpy  as np
from sklearn.datasets import load_breast_cancer

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

```

# 特征处理基础方法和实现

```python
x = np.array([1., 2., 3., 4., 5.])
x - x.mean()
```

    array([-2., -1.,  0.,  1.,  2.])


```python
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
x2 = s.fit_transform(x.reshape(-1, 1))
x2.mean(),x2.std()

```

    (0.0, 0.9999999999999999)


```python
from sklearn.preprocessing import MinMaxScaler

# feature_range 可以任意指定
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(x.reshape(-1, 1))
```

    array([[0.  ],
           [0.25],
           [0.5 ],
           [0.75],
           [1.  ]])

```python
from sklearn.preprocessing import normalize
X = np.array([[1, -1, 2], [2, 1, 0], [0, 1, -1]])
# norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
X_normalized = normalize(X, norm='l2')
X_normalized
```
    array([[ 0.40824829, -0.40824829,  0.81649658],
           [ 0.89442719,  0.4472136 ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678]])

```python
x = np.array([1,10,100,1000,10000])
x_log = np.log(x)
x_log
```
    array([0.        , 2.30258509, 4.60517019, 6.90775528, 9.21034037])

```python
x = pd.Series([1,2,3,4,5])
x2 = (x>3).astype(int)
x2.values
```
    array([0, 0, 0, 1, 1])

```python
from sklearn.preprocessing import LabelEncoder
x = ['b', 'b', 'a', 'c', 'b']
encoder = LabelEncoder()
x2 = encoder.fit_transform(x)
x2
```
    array([1, 1, 0, 2, 1])

```python
x2 = pd.Series(x).astype('category')
x2.cat.codes.values
```

    array([1, 1, 0, 2, 1], dtype=int8)

```python
import pandas as pd
x2, uniques = pd.factorize(x)
x2
```
    array([0, 0, 1, 2, 0])

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one_feature = ['b', 'a', 'c']
label_encoder = LabelEncoder()
feature = label_encoder.fit_transform(one_feature)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(feature.reshape(-1, 1))

```
    array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.]])

```python
from sklearn.preprocessing import LabelBinarizer
LabelBinarizer().fit_transform(one_feature)
```
    array([[0, 1, 0],
           [1, 0, 0],
           [0, 0, 1]])

```python
one_feature = ['b', 'a', 'c']
pd.get_dummies(one_feature,prefix='test')
```

| test_a | test_b | test_c |      |
| -----: | -----: | -----: | ---- |
|      0 |      0 |      1 | 0    |
|      1 |      1 |      0 | 0    |
|      2 |      0 |      0 | 1    |


```python
pd.get_dummies(one_feature,prefix='test',drop_first=True)
```


|      | test_b | test_c |
| ---: | -----: | -----: |
|    0 |      1 |      0 |
|    1 |      0 |      0 |
|    2 |      0 |      1 |



```python
x = pd.Series(['a', 'b', 'c', 'd', 'e', 'a', 'a', 'c'])
map_data_by_value_count(x, 2)
```

    split to 2+1 category

    {'a': 0, 'c': 1, 'd': 2, 'b': 2, 'e': 2}

```python

```

## 交叉统计
```python
np.random.seed(0)
test_df = pd.DataFrame({
    'x': np.random.choice(['红' ,'绿', '蓝'],1000),
    'y': np.random.randint(2, size=1000)
})

```

```python
pd.crosstab(test_df['y'], test_df['x'],margins=True)
```

|    x |   红 |   绿 |   蓝 |  All |
| ---: | ---: | ---: | ---: | ---: |
|    y |      |      |      |      |
|    0 |  153 |  165 |  163 |  481 |
|    1 |  184 |  170 |  165 |  519 |
|  All |  337 |  335 |  328 | 1000 |


```python
cal_woe(test_df['x'],test_df['y'])
```

    x
    红    0.108461
    绿   -0.046184
    蓝   -0.063841
    dtype: float64

## 日期衍生

```python
t = pd.Series([
    '2018-07-19T09:38:55.795+08:00', '2018-01-20T21:29:05.306+08:00',
    '2018-12-26T09:36:10.334+08:00', '2017-11-16T18:43:19.857+08:00',
    '2019-01-20T00:16:22.355+08:00', '2018-04-13T15:12:30.334+08:00',
])
DateTimeProcess(t).process()

```

|      |  Mth | PeriodOfMonth | isWeekend | PeriodOfDay |
| ---: | ---: | ------------: | --------: | ----------: |
|    0 |    7 |             2 |         0 |           2 |
|    1 |    1 |             2 |         1 |           4 |
|    2 |   12 |             3 |         0 |           2 |
|    3 |   11 |             2 |         0 |           3 |
|    4 |    1 |             2 |         1 |           1 |
|    5 |    4 |             2 |         0 |           3 |


# 离散化

```python
import sklearn
import pandas as pd
import numpy  as np
from sklearn.datasets import load_breast_cancer
```

## data


```python
bc = load_breast_cancer()
y = bc.target
X = pd.DataFrame.from_records(data=bc.data, columns=bc.feature_names)
X.shape
```

    (569, 30)

```python
# 转化为df
df = X
df['target'] = y
df.shape
```

    (569, 31)
    
## 基础离散化方法

```python
value,cutoff = pd.cut(df['mean radius'],bins=8,retbins=True,precision=2)
cutoff
```

    array([ 6.959871,  9.622125, 12.26325 , 14.904375, 17.5455  , 20.186625,
           22.82775 , 25.468875, 28.11    ])

```python
s1 = pd.Series([1,2,3,4,5,6])
value,cutoff = pd.qcut(s1,3,retbins=True)
value.value_counts()
```

    (4.333, 6.0]      2
    (2.667, 4.333]    2
    (0.999, 2.667]    2
    dtype: int64

```python
s2 = pd.Series([1,2,3,4,5,6,6,6,6])
value,cutoff = pd.qcut(s2,3,duplicates='drop',retbins=True)
value.value_counts(sort=False)
```

    (0.999, 3.667]    3
    (3.667, 6.0]      6
    dtype: int64

```python
np.array(np.percentile(df['mean radius'], [0,25, 50, 75,100])) 
```

    array([ 6.981, 11.7  , 13.37 , 15.78 , 28.11 ])

```python
value,cutoff = pd.qcut(df['mean radius'],4,retbins=True)
cutoff
```

    array([ 6.981, 11.7  , 13.37 , 15.78 , 28.11 ])
## 高级离散化方法

### 熵


```python
y= pd.Series([0,1,0,1,1])
```

```python
entropy.entropy(y)
```

    0.9709505944546686

```python
x = pd.Series([1,2,3,4,5])
entropy.info_gain(y, (x < 4).astype(int))

```

    0.4199730940219749
### sklearn决策树离散化


```python
cutoff  = dt_entropy_cut(df['mean radius'],df['target'] )
cutoff = cutoff.tolist()
[np.round(x,3) for x in cutoff]

```
    [10.945, 13.095, 13.705, 15.045, 17.8, 17.88]
    
### 最小熵离散化


```python
cut_by_entropy(df[['mean radius','target']],'target',margin=0.001)
```
    max_p=15.05,gain=0.4629862529990506
    max_p=13.11,gain=0.07679344919283099
    max_p=10.95,gain=0.02701401365980899
    max_p=13.71,gain=0.015068640676456302
    max_p=17.91,gain=0.08068906189021191
    max_p=17.85,gain=0.03769847832682838
    max_p=17.91,gain=-0.0

    [10.95, 13.11, 13.71, 15.05, 17.85, 17.91]

### Best-KS离散化

```python
df_ks = CalKS.cal_ks(
    df[['mean radius', 'target']], is_pivot=False, label='target')

```

    KS: 0.728621637334179
    
```python
bestks_cut (df[['mean radius','target']],'target')
```

    KS: 0.728621637334179
    KS: 0.47553282182438195
    KS: 0.35488813974869754
    KS: 0.16237402015677493
    KS: 0.6433747412008282
    KS: 0.24090909090909093
    KS: 0.829059829059829

    [11.75, 13.7, 13.08, 15.04, 16.84, 15.27, 17.85]
    
### 卡方离散化

减少了精度，便于演示和快速计算和查看

```python
x = df['mean radius'].round(0)

# 便于演示，先等分12箱
value,cutoff = \
pd.cut(x,bins=12,retbins=True,precision=0,include_lowest=True)

# 便于演示将初始分隔点取整
cutoff = cutoff.round(0)
```

再次分箱

```python
value,cutoff = \
    pd.cut(x,bins=cutoff,retbins=True,precision=0,include_lowest=True)
```

```python
freq_tab = pd.crosstab(value, df['target'])
```

```python
# 转化为numpy多维数组
freq = freq_tab.values
```

此处以95%的置信度（自由度为类数目-1）设定阈值。

```python
from scipy.stats import chi2
threshold = chi2.isf(0.05, df=1)
threshold
```

    3.8414588206941285

```python
cvs = np.array([])
for i in range(len(freq) - 1):
    cvs = np.append(cvs, stats_chi2(freq[i:i + 2]))

_c1 = lambda x: x < threshold
```
    Data Error
    Data Error
    Data Error
    Data Error
    Data Error
    Data Error
    
```python
while _c1(cvs.min()):
    cvs, freq, cutoff = chi2_merge_core(cvs, freq, cutoff, cvs.argmin())

```

    最小卡方值索引: 0 ；分割点: [ 7.  9. 10. 12. 14. 16. 18. 19. 21. 23. 24. 26. 28.]
    最小卡方值索引: 5 ；分割点: [ 7. 10. 12. 14. 16. 18. 19. 21. 23. 24. 26. 28.]
    Data Error
    最小卡方值索引: 5 ；分割点: [ 7. 10. 12. 14. 16. 18. 21. 23. 24. 26. 28.]
    Data Error
    最小卡方值索引: 5 ；分割点: [ 7. 10. 12. 14. 16. 18. 23. 24. 26. 28.]
    Data Error
    最小卡方值索引: 5 ；分割点: [ 7. 10. 12. 14. 16. 18. 24. 26. 28.]
    Data Error
    最小卡方值索引: 5 ；分割点: [ 7. 10. 12. 14. 16. 18. 26. 28.]
    最小卡方值索引: 4 ；分割点: [ 7. 10. 12. 14. 16. 18. 28.]
