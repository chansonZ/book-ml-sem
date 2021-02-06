# 第5章 数据分析与处理

## 数据分析工具

本章设计并开发了一款数据分析工具：该工具的输入可以是未做任何处理的原始数据，然后以报告形式输出，方便查阅和数据汇报。


# 使用示例

## 准备数据

```python
import seaborn as sns

# 取seaborn里的泰坦尼克数据集
titanic_df = sns.load_dataset('titanic')

# 数据示例：包含数值、类别等多种数据类型；空和非空等
titanic_df.head()
```

|      | survived | pclass |    sex |  age | sibsp | parch |    fare | embarked | class |   who | adult_male | deck | embark_town | alive | alone |
| ---: | -------: | -----: | -----: | ---: | ----: | ----: | ------: | -------: | ----: | ----: | ---------: | ---: | ----------: | ----: | ----: |
|    0 |        0 |      3 |   male | 22.0 |     1 |     0 |  7.2500 |        S | Third |   man |       True |  NaN | Southampton |    no | False |
|    1 |        1 |      1 | female | 38.0 |     1 |     0 | 71.2833 |        C | First | woman |      False |    C |   Cherbourg |   yes | False |
|    2 |        1 |      3 | female | 26.0 |     0 |     0 |  7.9250 |        S | Third | woman |      False |  NaN | Southampton |   yes |  True |
|    3 |        1 |      1 | female | 35.0 |     1 |     0 | 53.1000 |        S | First | woman |      False |    C | Southampton |   yes | False |
|    4 |        0 |      3 |   male | 35.0 |     0 |     0 |  8.0500 |        S | Third |   man |       True |  NaN | Southampton |    no |  True |

## 使用分析工具类：DataMetaInfo

输入参数为DataFrame：


```python
dm = DataMetaInfo(titanic_df)
```

输出，执行过程：

    na col row and unique stat
    factor stat
    constant_cols stat
    all na stat
    near zero var stat
    max len stat

查看数据基本统计：

```python
dm.psummary()
```
输出，执行过程：

    Data shape:
    (891, 15)
    
    Data mem size:
    0.079M
    
    Sum of duplicated rows:
    0
    
    Duplicated columns:
    []
    
    Nero zero var columns:0
    []
    
    All na columns:0
    []
    
    All Na rows:0
    []

**运行数据分析：**

```python
r1,r2,r3 = dm.run()
```

输出，执行过程：

    There is no zero_var columns~
    metrics of numeric stat

r1，r2，r3输出如下所示。其中：

- r1为空，表示该数据集中没有方差接近0的列
- r2显示了数值型变量的数理统计指标
- r3显示了各列的统计信息


```python
r2
```

输出如下：

|    | Pct1 | Pct5 | Pct25 |   Pct50 |   Pct75 | Pct95 |     Pct99 |       Min |  Max |     Mean |       Std |       Mad |  Skewness |  Kurtosis | NumOfNegative | PctOfNegative | NumOfPositive | PctOfPositive | NumOfZero | PctOfZero |
| ----:| ---: | ---: | ----: | ------: | ------: | ----: | --------: | --------: | ---: | -------: | --------: | --------: | --------: | --------: | ------------: | ------------: | ------------: | ------------: | --------: | -------- |
| survived |  0.0 | 0.000 |  0.0000 |  0.0000 |   1.0 |   1.00000 |   1.00000 | 0.00 |   1.0000 |  0.383838 |  0.486592 |  0.473013 |  0.478523 |     -1.775005 |             0 |           0.0 |           342 |  0.383838 |       549 | 0.616162 |
|   pclass |  1.0 | 1.000 |  2.0000 |  3.0000 |   3.0 |   3.00000 |   3.00000 | 1.00 |   3.0000 |  2.308642 |  0.836071 |  0.761968 | -0.630548 |     -1.280015 |             0 |           0.0 |           891 |  1.000000 |         0 | 0.000000 |
|      age |  1.0 | 4.000 | 20.1250 | 28.0000 |  38.0 |  56.00000 |  65.87000 | 0.42 |  80.0000 | 29.699118 | 14.526497 | 11.322944 |  0.389108 |      0.178274 |             0 |           0.0 |           714 |  0.801347 |         0 | 0.000000 |
|    sibsp |  0.0 | 0.000 |  0.0000 |  0.0000 |   1.0 |   3.00000 |   5.00000 | 0.00 |   8.0000 |  0.523008 |  1.102743 |  0.713780 |  3.695352 |     17.880420 |             0 |           0.0 |           283 |  0.317621 |       608 | 0.682379 |
|    parch |  0.0 | 0.000 |  0.0000 |  0.0000 |   0.0 |   2.00000 |   4.00000 | 0.00 |   6.0000 |  0.381594 |  0.806057 |  0.580742 |  2.749117 |      9.778125 |             0 |           0.0 |           213 |  0.239057 |       678 | 0.760943 |
|     fare |  0.0 | 7.225 |  7.9104 | 14.4542 |  31.0 | 112.07915 | 249.00622 | 0.00 | 512.3292 | 32.204208 | 49.693429 | 28.163692 |  4.787317 |     33.398141 |             0 |           0.0 |           876 |  0.983165 |        15 | 0.016835 |




```python
r3
```

|      | types |  dtypes | NumOfNan | PctOFNan | NumOfUnique | IsAllNa | NearZeroVar | MaxLenOfStrValue |
| -----: | -----: | ------: | -------: | -------: | ----------: | ------: | ----------: | ---------------: |
|    survived |  factor |    int64 |        0 |    0.000000 |       2 |       False |            False | NaN  |
|      pclass |  factor |    int64 |        0 |    0.000000 |       3 |       False |            False | NaN  |
|         sex |  factor |   object |        0 |    0.000000 |       2 |       False |            False | 6.0  |
|         age | numeric |  float64 |      177 |    0.198653 |      88 |       False |            False | NaN  |
|       sibsp |  factor |    int64 |        0 |    0.000000 |       7 |       False |            False | NaN  |
|       parch |  factor |    int64 |        0 |    0.000000 |       7 |       False |            False | NaN  |
|        fare | numeric |  float64 |        0 |    0.000000 |     248 |       False |            False | NaN  |
|    embarked |  factor |   object |        2 |    0.002245 |       3 |       False |            False | 1.0  |
|       class |  factor | category |        0 |    0.000000 |       3 |       False |            False | 6.0  |
|         who |  factor |   object |        0 |    0.000000 |       3 |       False |            False | 5.0  |
|  adult_male |  factor |     bool |        0 |    0.000000 |       2 |       False |            False | NaN  |
|        deck |  factor | category |      688 |    0.772166 |       7 |       False |            False | 1.0  |
| embark_town |  factor |   object |        2 |    0.002245 |       3 |       False |            False | 11.0 |
|       alive |  factor |   object |        0 |    0.000000 |       2 |       False |            False | 3.0  |
|       alone |  factor |     bool |        0 |    0.000000 |       2 |       False |            False | NaN  |


