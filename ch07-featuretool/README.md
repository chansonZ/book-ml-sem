# 第7章 基于 Featuretools 的自动特征衍生

相关数据和代码

# featuretools 参考

https://docs.featuretools.com/api_reference.html

上述链接已更新为：https://featuretools.alteryx.com/en/stable/api_reference.html


# 请参考书中操作

## 聚合基元

```python
import featuretools as ft
 
primitives = ft.list_primitives()
primitives[primitives['type'] == 'aggregation'].head(6)
```



|      |            name |        type | dask_compatible | description                                       |
| ---: | --------------: | ----------: | --------------: | ------------------------------------------------- |
|    0 |   n_most_common | aggregation |           False | Determines the `n` most common elements.          |
|    1 |           trend | aggregation |           False | Calculates the trend of a variable over time.     |
|    2 |        num_true | aggregation |            True | Counts the number of `True` values.               |
|    3 | time_since_last | aggregation |           False | Calculates the time elapsed since the last dat... |
|    4 |             max | aggregation |            True | Calculates the highest value, ignoring `NaN` v... |
|    5 |             any | aggregation |            True | Determines if any value is 'True' in a list.      |

## 转换基元

|      |               name |      type | dask_compatible | description                                       |
| ---: | -----------------: | --------: | --------------: | ------------------------------------------------- |
|   22 |             negate | transform |            True | Negates a numeric value.                          |
|   23 |               hour | transform |            True | Determines the hour value of a datetime.          |
|   24 |                and | transform |            True | Element-wise logical AND of two lists.            |
|   25 | add_numeric_scalar | transform |            True | Add a scalar to each value in the list.           |
|   26 |       greater_than | transform |            True | Determines if values in one list are greater t... |
|   27 |               isin | transform |            True | Determines whether a value is present in a pro... |


## 数据类型


| Index | 数据类型 | 解释 |
| :----: | :----: | :----: |
|1|Index|表示唯一标识实体实例的变量|
|2|Id|表示一个实体实例的变量|
|3|TimeIndex|表示实体的时间索引
|4|DatetimeTimeIndex|代表日期时间的实体的时间索引
|5|NumericTimeIndex|表示数字实体的时间索引|
|6|Datetime|表示作为时间的变量|
|7|Numeric|表示包含数值的变量|
|8|Categorical|表示可以采用无序离散值的变量|
|9|Ordinal|表示采用有序离散值的变量|
|10|Boolean|表示布尔值的变量|
|11|Text|代表变量字符串|
|12|LatLong|表示一个数组（纬度，经度）|
|13|ZIPCode|代表美国的邮政地址|
|14|IPAddress|代表计算机网络IP地址|
|15|FullName|代表一个人的全名|
|16|EmailAddress|电子邮箱地址|
|17|URL|代表有效的网址|
|18|PhoneNumber|代表任何有效的电话号码|
|19|DateOfBirth|将出生日期表示为日期时间|
|20|CountryCode|代表ISO-3166标准国家代码|
|21|SubRegionCode|表示一个ISO-3166标准子区域代码|
|22|FilePath|表示有效的文件路径|



# 捷信数据

数据下载：https://www.kaggle.com/c/home-credit-default-risk/data


```python
import pandas as pd
import numpy as np
import featuretools as ft

#使用pandas读取7个数据表生成dataframe
app_train = pd.read_csv('input/application_train.csv')
app_test = pd.read_csv('input/application_test.csv')
bureau = pd.read_csv('input/bureau.csv')
bureau_balance = pd.read_csv('input/bureau_balance.csv')
cash = pd.read_csv('input/POS_CASH_balance.csv')
credit = pd.read_csv('input/credit_card_balance.csv')
previous = pd.read_csv('input/previous_application.csv')
installments =pd.read_csv('input/installments_payments.csv')

app_test['TARGET'] = np.nan
app = app_train.append(app_test, ignore_index = True, sort = True)

#有几张表的索引字段数据类型是浮点类型，需要把它们转换为整型，以确保可以正常地添加关系。
for index in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
    for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
        if index in list(dataset.columns):
            dataset[index] = dataset[index].fillna(0).astype(np.int64)
            
            
```

## 构建实体和实体集

```python
es = ft.EntitySet(id = 'clients')

import featuretools.variable_types as vtypes

app_types = {}

# 将两种类别的变量调整为布尔型
for col in app:
    if (app[col].nunique() == 2) and (app[col].dtype == float):
        app_types[col] = vtypes.Boolean

# 剔除目标变量 `TARGET`
del app_types['TARGET']


#某些类别变量的业务含义上还有排序性，需要配置为Ordinal类型。
app_types['REGION_RATING_CLIENT'] = vtypes.Ordinal
app_types['REGION_RATING_CLIENT_W_CITY'] = vtypes.Ordinal
app_types['HOUR_APPR_PROCESS_START'] = vtypes.Ordinal

previous_types = {}
for col in previous:
    if (previous[col].nunique() == 2) and (previous[col].dtype == float):
        previous_types[col] = vtypes.Boolean
        
installments = installments.drop(columns = ['SK_ID_CURR'])
credit = credit.drop(columns = ['SK_ID_CURR'])
cash = cash.drop(columns = ['SK_ID_CURR'])

# 创建具有唯一标识索引的实体
es = es.entity_from_dataframe(entity_id='app',
                              dataframe=app,
                              index='SK_ID_CURR',
                              variable_types=app_types)

es = es.entity_from_dataframe(entity_id='bureau',
                              dataframe=bureau,
                              index='SK_ID_BUREAU')

es = es.entity_from_dataframe(entity_id='previous',
                              dataframe=previous,
                              index='SK_ID_PREV',
                              variable_types=previous_types)

# 创建没有唯一标识索引的实体
es = es.entity_from_dataframe(entity_id='bureau_balance',
                              dataframe=bureau_balance,
                              make_index=True,
                              index='bureaubalance_index')

es = es.entity_from_dataframe(entity_id='cash',
                              dataframe=cash,
                              make_index=True,
                              index='cash_index')

es = es.entity_from_dataframe(entity_id='installments',
                              dataframe=installments,
                              make_index=True,
                              index='installments_index')

es = es.entity_from_dataframe(entity_id='credit',
                              dataframe=credit,
                              make_index=True,
                              index='credit_index')

es
# 输出实体集信息  
```

## 构建关系

```python

print('Parent: app, Parent Variable of bureau: SK_ID_CURR\n\n',
      app.iloc[:, 111:114].head())  #111:115
print(
    '\nChild: bureau, Child Variable of app: SK_ID_CURR\n\n',
    bureau[bureau['SK_ID_CURR'] == 100002].iloc[:, :3])
# 输出

```

```python
Parent: app, Parent Variable of bureau: SK_ID_CURR

    SK_ID_CURR  TARGET  TOTALAREA_MODE
0      100002     1.0          0.0149
1      100003     0.0          0.0714
2      100004     0.0             NaN
3      100006     0.0             NaN
4      100007     0.0             NaN

Child: bureau, Child Variable of app: SK_ID_CURR

        Unnamed: 0  SK_ID_CURR  SK_ID_BUREAU
37884      675684      100002       6158904
37885      675685      100002       6158905
37886      675686      100002       6158906
37887      675687      100002       6158907
37888      675688      100002       6158908
37889      675689      100002       6158909
75006     1337779      100002       6158903
83119     1486113      100002       6113835
```

## 添加到EntitySet中

```python

# 为app 和bureau构建关联关系
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'],
                               es['bureau']['SK_ID_CURR'])

# 为bureau和bureau _balance构建关联关系
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'],
                                   es['bureau_balance']['SK_ID_BUREAU'])

# 为app 和previous构建关联关系
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'],
                                 es['previous']['SK_ID_CURR'])

# 为previous与 cash、 installments、credit 构建关联关系
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'],
                                  es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'],
                                          es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'],
                                    es['credit']['SK_ID_PREV'])

# 构建好的关系添加到实体集
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])

es


Entityset: clients
  Entities:
    app [Rows: 20001, Columns: 123]
    bureau [Rows: 95905, Columns: 18]
    previous [Rows: 93147, Columns: 38]
    bureau_balance [Rows: 1376507, Columns: 5]
    cash [Rows: 556515, Columns: 9]
    installments [Rows: 748735, Columns: 9]
    credit [Rows: 208716, Columns: 24]
  Relationships:
    bureau.SK_ID_CURR -> app.SK_ID_CURR
    bureau_balance.SK_ID_BUREAU -> bureau.SK_ID_BUREAU
    previous.SK_ID_CURR -> app.SK_ID_CURR
    cash.SK_ID_PREV -> previous.SK_ID_PREV
    installments.SK_ID_PREV -> previous.SK_ID_PREV
    credit.SK_ID_PREV -> previous.SK_ID_PREV
    
```

# 特征基元

```python

# 自定义函数
def plusOne(column):
    return column + 1
# 通过接口添加自定义基元
plus_one = ft.primitives.make_trans_primitive(
    function=plusOne,
    input_types=[ft.variable_types.Numeric],
    return_type=ft.variable_types.Numeric)

feature_matrixp, feature_namesp = ft.dfs(entityset=es,
                                         target_entity='app',
                                         trans_primitives=[plus_one],
                                         agg_primitives=['count'],
                                         max_depth=2)
# 查看两组比对衍生特征
feature_matrixp[['AMT_ANNUITY','PLUSONE(AMT_ANNUITY)','COUNT(bureau)','PLUSONE(COUNT(bureau))']]

```

|            | AMT_ANNUITY | PLUSONE(AMT_ANNUITY) | COUNT(bureau) | PLUSONE(COUNT(bureau)) |
| ---------: | ----------: | -------------------: | ------------: | ---------------------: |
| SK_ID_CURR |             |                      |               |                        |
|     100002 |     24700.5 |              24701.5 |           8.0 |                    9.0 |
|     100003 |     35698.5 |              35699.5 |           4.0 |                    5.0 |
|     100004 |      6750.0 |               6751.0 |           2.0 |                    3.0 |
|     100006 |     29686.5 |              29687.5 |           0.0 |                    1.0 |
|        ... |         ... |                  ... |           ... |                    ... |


# 深度特征合成

# 使用dfs默认基元合成特征

```python
feature_names = ft.dfs(entityset=es,
                       target_entity='app',
                       features_only=True)
print (len(feature_names))
feature_names[1000:1010]

[<Feature: STD(previous.MIN(credit.AMT_DRAWINGS_OTHER_CURRENT))>,
 <Feature: STD(previous.SUM(cash.CNT_INSTALMENT))>,
 <Feature: STD(previous.SKEW(installments.NUM_INSTALMENT_VERSION))>,
 <Feature: STD(previous.SUM(credit.AMT_DRAWINGS_POS_CURRENT))>,
 <Feature: STD(previous.MAX(credit.CNT_DRAWINGS_OTHER_CURRENT))>,
 <Feature: STD(previous.MEAN(credit.CNT_DRAWINGS_ATM_CURRENT))>,
 <Feature: STD(previous.SUM(installments.DAYS_ENTRY_PAYMENT))>,
 <Feature: STD(previous.MIN(credit.CNT_DRAWINGS_CURRENT))>,
 <Feature: STD(previous.MIN(cash.MONTHS_BALANCE))>,
 <Feature: STD(previous.SUM(credit.SK_DPD))>]
```

 ```python
 
 # 选择运算基元
agg_primitives = [
    "sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"
]
trans_primitives = ['percentile', 'and']

# 调用dfs接口 
feature_matrix, feature_names = ft.dfs(entityset=es,
                                       target_entity='app',
                                       agg_primitives=agg_primitives,
                                       trans_primitives=trans_primitives,
                                       features_only=False,
                                       max_depth=2)
 
 ```


