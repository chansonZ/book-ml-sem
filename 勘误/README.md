# 第4章 94页公式(4-6)错误



错误：
![error](./imgs/e1.png)


更正为：
![right](./imgs/r1.png)




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

