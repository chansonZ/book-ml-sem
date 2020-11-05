
from sklearn.tree import DecisionTreeClassifier
# max_depth=3，表示进行3次划分构造3层的树结构
def dt_entropy_cut(x, y, max_depth=3, criterion='entropy'):  # gini
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dt.fit(x.values.reshape(-1, 1), y)
    qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]
    if qts.shape[0] == 0:
        qts = np.array([np.median(data[:, feature])])
    else:
        qts = np.sort(qts)
    return qts


def cut_by_entropy(df, label, loop=3, margin=0.01):
    '''停止准则：
    1、已到循环次数
    2、信息增益增加小于阈值（用最小熵，不便定义熵值的大小）
    '''
    assert len(df.columns) == 2, 'not support'

    def _get_best_points(df, label, feature_col, loop=3, margin=0.01):
        if loop == 0:
            return [None]
        else:
            # 计算信息增益
            gain, max_p = get_max_gain_point(df, label, feature_col)
            print('max_p={},gain={}'.format(max_p, gain))
            # 信息增益小于指定阈值时停止分箱
            if gain < margin:
                return [None]

            # 左闭，右开
            left = df.loc[df[feature_col] < max_p, :]
            right = df.loc[df[feature_col] >= max_p, :]

            # 递归分箱
            return [max_p] + \
                _get_best_points(left, label, feature_col,loop - 1,  margin) + \
                _get_best_points(right, label, feature_col, loop - 1, margin)

    feature_col = [aa for aa in df.columns.tolist() if aa != label][0]
    points = _get_best_points(df, label, feature_col, loop, margin)
    points = [p for p in points if p is not None]
    points = list(set(points))
    points.sort()
    return points


def get_max_gain_point(df, label, feature_col):
    '''
    分箱后，使用离散方法计算信息增益
    注意这里的for实现效率较低，实践中请先粗分类
    '''
    gain, max_p = -1, -1
    ps = df[feature_col].unique().tolist()
    ps.sort()

    if len(ps) < 2:
        return -1, None

    for pp in ps:
        #tmp = (df[feature_col] <= pp).astype(int)
        tmp = (df[feature_col] < pp).astype(int)
        g = entropy.info_gain(df[label], tmp)
        if g > gain:
            gain = g
            max_p = pp
    return gain, max_p
    
    
class CalKS:
    @staticmethod
    def pivot(df, label):
        """
        1、计算数据透视表-pivot
        2、只支持二分类
        paramters
        ----------
        df: 带标签的DataFrame,共两列数据
        label:带label的列名

        return
        ------
        DataFrame—数据透视表
        """
        assert len(df.columns) == 2, 'not support'
        df_ = df.copy()
        feature_col = [aa for aa in df_.columns.tolist() if aa != label][0]

        return pd.pivot_table(
            df_,
            index=[feature_col],
            columns=[label],
            aggfunc=len,
            fill_value=0)

    @staticmethod
    def cal_ks(df, is_pivot=True, label=None):
        '''
        paramters
        ----------
        df:    待计算的两列数据：x,y
        label：label列名
        计算方法：
            abs(表签0累计用户占比 - 表签1累计用户占比)
        return
        ------
        DataFrame—KS统计表
        '''
        pivot_df = df.copy()
        if is_pivot is False:
            pivot_df = CalKS.pivot(df, label)

        # 二分类（0,1）实际就是  [0, 1]
        label_value = pivot_df.columns.tolist()

        # 表签0的数量
        count_0 = pivot_df[label_value[0]].sum()
        pivot_df['cum_percent_1'] = \
        pivot_df[label_value[0]].cumsum() / count_0

        # 表签1的数量
        count_1 = pivot_df[label_value[1]].sum()
        pivot_df['cum_percent_2'] = \
        pivot_df[label_value[1]].cumsum() / count_1

        pivot_df['KS'] = pivot_df['cum_percent_1'].sub(
            pivot_df['cum_percent_2']).abs()
        print('KS:', pivot_df["KS"].max())
        return pivot_df


def bestks_cut (df,
                           label,
                           loop=3,
                           min_count=0.01,
                           return_combine=True):
    # 去除了空值
    pivot_df = CalKS.pivot(df, label)
    if 0.0 < min_count < 1.0:
        min_count = int(min_count * df.shape[0])
    min_count = max(min_count, int(0.01 * df.shape[0]))
    assert min_count > 1, 'bestks_cut:wrong min_count:{}'.format(min_count)
    return get_bestks_points (pivot_df, loop, min_count)

def get_bestks_points (df, loop=3, min_count=5):
    cols = df.columns.tolist()
    # 是否为空箱
    def _is_null(df):
        return df.shape[0] == 0

    # 是否全0或全1的标签
    def _is_only_one_class(df, cols):
        return df[cols[0]].sum() == 0 or df[cols[1]].sum() == 0

    # 箱内的样本数量是否太少
    def _is_too_small(df, cols, min_count):
        return (df[cols[0]].sum() + df[cols[1]].sum()) < min_count
        
    def _split_bestks(df, loop=3, min_count=5):
        '''
        df:透视表格式
        '''
        if loop == 0 or _is_null(df) or _is_only_one_class(
                df, cols) or _is_too_small(df, cols, min_count):
            return [None]
        else:
            max_p = CalKS.cal_ks(df).idxmax()['KS']
            left = df.loc[df.index < max_p, :]
            right = df.loc[df.index >= max_p, :]

            # 如果这个max_p
            if _is_null(left) or _is_null(right) or \
                _is_only_one_class(left, cols) or \
                _is_only_one_class(right, cols) or \
                _is_too_small(left, cols, min_count) or \
                _is_too_small(right, cols, min_count):
                return [None]
            
            # 左右子树递归
            return [max_p] + _split_bestks(left, loop - 1,min_count) \
                            + _split_bestks(right, loop - 1, min_count)

    points = _split_bestks(df, loop, min_count)
    points = [p for p in points if p is not None]
    points = list(set(points))
    return points


from scipy import stats
def stats_chi2(arr,correction=False):
    try:
        # 便于演示，此处统一未使用校准，实际中可根据频数精确控制
        s=stats.chi2_contingency(arr,correction=correction)
    except ValueError:
    # 返回0认为0的组应该进行合并
        print('Data Error')
        return 0
    return s[0]
    
def chi2_merge_core(cvs, freq, cutoffs, minidx):
    '''卡方合并逻辑'''
    print('最小卡方值索引:',minidx,'；分割点:',cutoffs)
    # minidx后一箱合并到前一组
    tmp = freq[minidx] + freq[minidx + 1]
    freq[minidx] = tmp
    # 删除minidx后一组
    freq = np.delete(freq, minidx + 1, 0)
    # 删除对应的分隔点
    cutoffs = np.delete(cutoffs, minidx + 1, 0)
    cvs = np.delete(cvs, minidx, 0)

    # 更新前后两个组的卡方值，其他部分卡方值未变化
    if minidx <= (len(cvs) - 1):
        cvs[minidx] = stats_chi2(freq[minidx:minidx + 2])
    if minidx - 1 >= 0:
        cvs[minidx - 1] = stats_chi2(freq[minidx - 1:minidx + 1])
    return cvs, freq, cutoffs
