# 第16章  模型稳定性监控

相关数据和代码

# 使用示例

## 准备数据

```python
rndm = np.random.RandomState(42)
e = rndm.normal(size=10**2)
a = rndm.normal(size=10**2)
```

## 计算PSI

```python
psi_d = psi(pd.Series(e),pd.Series(a))

psi_d

{'data':                   expect_pct  actual_pct
 (-inf, -1.328]          0.11        0.06
 (-1.328, -0.72]         0.10        0.22
 (-0.72, -0.502]         0.10        0.04
 (-0.502, -0.301]        0.10        0.04
 (-0.301, -0.116]        0.10        0.05
 (-0.116, 0.111]         0.09        0.10
 (0.111, 0.331]          0.10        0.14
 (0.331, 0.648]          0.10        0.12
 (0.648, 1.004]          0.10        0.11
 (1.004, inf]            0.10        0.12, 'statistic': 0.2922923789663523}
```

## 查看数据

```python
psi_d['data']

expect_pct	actual_pct
(-inf, -1.328]	0.11	0.06
(-1.328, -0.72]	0.10	0.22
(-0.72, -0.502]	0.10	0.04
(-0.502, -0.301]	0.10	0.04
(-0.301, -0.116]	0.10	0.05
(-0.116, 0.111]	0.09	0.10
(0.111, 0.331]	0.10	0.14
(0.331, 0.648]	0.10	0.12
(0.648, 1.004]	0.10	0.11
(1.004, inf]	0.10	0.12
```

```python
psi_d['statistic']

0.30907897278111285
```

## 封装，计算整个dataframe的PSI

```python
# 准备数据
rndm = np.random.RandomState(42)
e1 = rndm.normal(size=10**2)
a1 = rndm.normal(size=10**2)
e2 = rndm.normal(size=10**2)
a2 = rndm.normal(size=10**2)
e = pd.DataFrame({'col1': e1, 'col2': e2})
a = pd.DataFrame({'col1': a1, 'col2': a2})
```

### 计算PSI

```python
psi_df = cal_df_psi(e, a)

psi_df
```

​      column	psi
0	col1	0.292292
1	col2	0.184779


## 模拟空值计算PSI


```python
e = rndm.normal(size=10**2)
a = rndm.normal(size=10**2)

e[2:10] = np.NaN
a[20:25] = np.NaN

psi_d = psi(pd.Series(e),pd.Series(a))
```

Nan statistics for expect:
8
Nan statistics for actual:
5
    Drop Nan !!

```python
psi_d
```

{'data':                   expect_pct  actual_pct
 (-2.698, -1.251]    0.108696    0.136842
 (-1.251, -0.922]    0.097826    0.052632
 (-0.922, -0.427]    0.097826    0.157895
 (-0.427, -0.26]     0.097826    0.010526
 (-0.26, -0.0543]    0.108696    0.073684
 (-0.0543, 0.129]    0.097826    0.094737
 (0.129, 0.593]      0.097826    0.200000
 (0.593, 0.823]      0.097826    0.084211
 (0.823, 1.345]      0.097826    0.115789
 (1.345, 2.573]      0.097826    0.063158, 'statistic': 0.36488760566199535}