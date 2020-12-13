
def cal_A_B(pdo=20, base_score=500, odds=1 / 50):
    B = pdo / np.log(2)
    A = base_score + B * np.log(odds)
    return A, B
    
'''
parameter
---------
df:变量的woe,要求与模型训练logit时的列顺序一样
logit：sklearn中的逻辑回归模型,带截距

return
------
    新增每行数据的评分列：Score

example:
    df= cal_score(df,logit)
'''

def cal_score_byadd(df, logit, A=387.123, B=28.854):
    def _cal_woe_score(x, beta, n, B, beta0, A):
        ''' 只计算总分'''
        score = 0.0
        for cc in x.index.tolist():
            score += x[cc] * beta[cc]
        score = A - B * (beta0 + score)
        return score

    beta = dict(zip(df.columns.tolist(), logit.coef_[0]))

    n = df.shape[1]
    beta0 = logit.intercept_[0]

    df['Score'] = df.apply(lambda x: _cal_woe_score(x, beta, n, B, beta0, A),
                           axis=1)
    return df


def cal_score_byodds(df, logit, A=387.123, B=28.854):
    beta0 = logit.intercept_[0]

    prob_01 = logit.predict_proba(df)
    df['Score'] = A - B * np.log(prob_01[:, 1] / prob_01[:, 0])
    return df
    
    