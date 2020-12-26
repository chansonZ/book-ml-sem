# 第4章 94页公式(4-6)错误



错误：

<img src="./imgs/e1.png" width = "355" height = "65" alt="" align=center />


更正为：

<img src="./imgs/r1.png" width = "379" height = "71" alt="" align=center />



## 第5章 138页—代码对齐问题



错误：

```python
@staticmethod
    def all_na_cols(df, index=False):
        ''' 全是缺失值的列 ''' 
    		if index:
            return DFutils.count_na_col(df) == df.shape[0]
        else:
            return DFutils.ser_index(DFutils.count_na_col(df) == df.shape[0])
      
```

更正为：

```python
		@staticmethod
    def all_na_cols(df, index=False):
				''' 全是缺失值的列 ''' 
    		if index:
            return DFutils.count_na_col(df) == df.shape[0]
        else:
            return DFutils.ser_index(DFutils.count_na_col(df) == df.shape[0])
```

