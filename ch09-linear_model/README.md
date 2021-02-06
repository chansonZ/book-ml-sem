# 第9章 线性模型

相关数据和代码


# ridge回归解析示例-其他请参考src文件夹


```python
from sklearn.linear_model import Ridge

def ridge_regression(data, predictors, alpha, models_to_plot):
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors], data['y'])
    y_pred = ridgereg.predict(data[predictors])

    # 绘制指定的alpha的图形
    if alpha in models_to_plot:
        myplot(data['x'], data['y'], y_pred, \
               models_to_plot[alpha], 'alpha=%.3g' % alpha)

    # 记录模型拟合效果rss、截距和系数
    return summary(data['y'], y_pred, ridgereg.intercept_, ridgereg.coef_)
```


```python
# 拟合了所有的 x
predictors = ['x']
predictors.extend(['x_%d' % i for i in range(2, pow_max)])

# 设置正则系数
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50]
```


```python
ind = ['alpha_%.2g' % alpha_ridge[i] for i in range(0, len(alpha_ridge))]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15: 221, 1e-3: 222, 1: 223, 50: 224}
for i in range(len(alpha_ridge)):
    coef_matrix_ridge.iloc[i, ] = ridge_regression(data, predictors,
                                                   alpha_ridge[i],
                                                   models_to_plot)
```


![png](./imgs/output_18_0.png)



```python
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_1e-15</th>
      <td>0.62</td>
      <td>0.9</td>
      <td>-0.12</td>
      <td>0.56</td>
      <td>1.2</td>
      <td>-2.9</td>
      <td>-3.8</td>
      <td>1.4</td>
      <td>3.7</td>
      <td>1.2</td>
      <td>-0.53</td>
      <td>-0.46</td>
      <td>-0.11</td>
      <td>-0.0096</td>
    </tr>
    <tr>
      <th>alpha_1e-10</th>
      <td>0.66</td>
      <td>0.94</td>
      <td>-0.014</td>
      <td>-0.095</td>
      <td>0.24</td>
      <td>-1.1</td>
      <td>-1.3</td>
      <td>0.31</td>
      <td>1.1</td>
      <td>0.65</td>
      <td>0.1</td>
      <td>-0.024</td>
      <td>-0.01</td>
      <td>-0.001</td>
    </tr>
    <tr>
      <th>alpha_1e-08</th>
      <td>0.71</td>
      <td>0.96</td>
      <td>0.018</td>
      <td>-0.58</td>
      <td>-0.29</td>
      <td>-0.021</td>
      <td>0.29</td>
      <td>0.24</td>
      <td>0.08</td>
      <td>0.013</td>
      <td>0.0004</td>
      <td>-0.00022</td>
      <td>0.00011</td>
      <td>3.4e-05</td>
    </tr>
    <tr>
      <th>alpha_0.0001</th>
      <td>0.74</td>
      <td>0.95</td>
      <td>-0.047</td>
      <td>-0.45</td>
      <td>0.079</td>
      <td>0.034</td>
      <td>-0.0081</td>
      <td>0.0024</td>
      <td>-0.00012</td>
      <td>-0.00012</td>
      <td>7.1e-05</td>
      <td>-2.1e-05</td>
      <td>2e-06</td>
      <td>2.3e-06</td>
    </tr>
    <tr>
      <th>alpha_0.001</th>
      <td>0.78</td>
      <td>0.94</td>
      <td>0.011</td>
      <td>-0.42</td>
      <td>0.018</td>
      <td>0.01</td>
      <td>-0.0047</td>
      <td>0.0014</td>
      <td>-0.00026</td>
      <td>2.3e-05</td>
      <td>9.2e-06</td>
      <td>-6.1e-06</td>
      <td>2.2e-06</td>
      <td>-4.9e-07</td>
    </tr>
    <tr>
      <th>alpha_0.01</th>
      <td>0.99</td>
      <td>0.87</td>
      <td>0.091</td>
      <td>-0.29</td>
      <td>0.018</td>
      <td>0.00085</td>
      <td>-0.0022</td>
      <td>0.00084</td>
      <td>-0.00025</td>
      <td>5.8e-05</td>
      <td>-8.8e-06</td>
      <td>-6.7e-07</td>
      <td>1.4e-06</td>
      <td>-7.9e-07</td>
    </tr>
    <tr>
      <th>alpha_0.1</th>
      <td>1.9</td>
      <td>0.77</td>
      <td>0.2</td>
      <td>-0.12</td>
      <td>0.018</td>
      <td>-0.0033</td>
      <td>0.00023</td>
      <td>7.7e-05</td>
      <td>-5.4e-05</td>
      <td>2.1e-05</td>
      <td>-7.1e-06</td>
      <td>2.1e-06</td>
      <td>-5.7e-07</td>
      <td>1.4e-07</td>
    </tr>
    <tr>
      <th>alpha_1</th>
      <td>4.6</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>-0.049</td>
      <td>0.012</td>
      <td>-0.0028</td>
      <td>0.00064</td>
      <td>-0.00013</td>
      <td>2.4e-05</td>
      <td>-2.8e-06</td>
      <td>-3.8e-07</td>
      <td>4.6e-07</td>
      <td>-2.4e-07</td>
      <td>1e-07</td>
    </tr>
    <tr>
      <th>alpha_5</th>
      <td>9.8</td>
      <td>0.41</td>
      <td>0.056</td>
      <td>-0.021</td>
      <td>0.0058</td>
      <td>-0.0016</td>
      <td>0.00046</td>
      <td>-0.00013</td>
      <td>3.6e-05</td>
      <td>-1e-05</td>
      <td>2.8e-06</td>
      <td>-8e-07</td>
      <td>2.3e-07</td>
      <td>-6.4e-08</td>
    </tr>
    <tr>
      <th>alpha_10</th>
      <td>13</td>
      <td>0.34</td>
      <td>0.035</td>
      <td>-0.014</td>
      <td>0.004</td>
      <td>-0.0012</td>
      <td>0.00035</td>
      <td>-0.0001</td>
      <td>3e-05</td>
      <td>-9e-06</td>
      <td>2.7e-06</td>
      <td>-8e-07</td>
      <td>2.4e-07</td>
      <td>-7.3e-08</td>
    </tr>
    <tr>
      <th>alpha_20</th>
      <td>16</td>
      <td>0.28</td>
      <td>0.021</td>
      <td>-0.0083</td>
      <td>0.0025</td>
      <td>-0.00076</td>
      <td>0.00023</td>
      <td>-7e-05</td>
      <td>2.1e-05</td>
      <td>-6.6e-06</td>
      <td>2e-06</td>
      <td>-6.2e-07</td>
      <td>1.9e-07</td>
      <td>-5.9e-08</td>
    </tr>
    <tr>
      <th>alpha_50</th>
      <td>20</td>
      <td>0.22</td>
      <td>0.0095</td>
      <td>-0.0039</td>
      <td>0.0012</td>
      <td>-0.00038</td>
      <td>0.00012</td>
      <td>-3.6e-05</td>
      <td>1.1e-05</td>
      <td>-3.5e-06</td>
      <td>1.1e-06</td>
      <td>-3.4e-07</td>
      <td>1.1e-07</td>
      <td>-3.3e-08</td>
    </tr>
  </tbody>
</table>
</div>
