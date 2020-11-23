# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第6章 特征工程


def map_data_by_value_count(ser, count):
    ''' 小于count的统一为一组 '''
    assert count > 0, 'count must larger than 0!'
    vc = ser.value_counts()
    up_list = vc[vc >= count].index.tolist()
    low_list = vc[vc < count].index.tolist()
    print('split to {}+{} category'.format(
        len(up_list), 1 if len(low_list) > 0 else 0))
    map_data = {}
    i = 0
    for mm in up_list:
        map_data.update({mm: i})
        i += 1
    for mm in low_list:
        map_data.update({mm: i})
    return map_data


def cal_woe(x, y):
    '''
    x,y：pd.Serises
        变量x为类别变量,y为0，1
    '''
    t = pd.crosstab(y, x)
    w = t.div(t.sum(axis=1), axis=0)
    # 坏样本分布 / 好样本分布
    return np.log(w.iloc[1, :] / w.iloc[0, :])


class DateTimeProcess:
    def __init__(self, s):
        # 格式化
        self.s = pd.to_datetime(s, errors='coerce')
        self.df = pd.DataFrame()
    
    def date_process(self):
        '''衍生日期特征
        return:
            Mth: 月份
            isWeekend: 是否周末。0：否,1：是
            PeriodOfMonth: 1：上旬，2：中旬，3：下旬
        '''
        def _level(d):
            if d < (1 / 3.0):
                return 1
            elif d > (2 / 3.0):
                return 3
            else:
                return 2
        self.df['Mth'] = self.s.dt.month
        t = self.s.dt.day * 1.0 / self.s.dt.daysinmonth
        self.df['PeriodOfMonth'] = t.apply(_level)
        # 数据在0~6之间，星期一是0，星期日是6
        self.df['isWeekend'] = (self.s.dt.dayofweek >= 5).apply(int)
    
    def time_process(self):
        '''衍生时间特征
        return:
            0：深夜：1：上午；2：下午；3：晚上
        '''
        def _hour(hour):
            if (hour >= 0) & (hour <= 6):
                return 1
            elif (hour > 6) & (hour <= 12):
                return 2
            elif (hour > 12) & (hour <= 18):
                return 3
            else:
                return 4
        self.df['PeriodOfDay'] = self.s.dt.hour.apply(_hour)

    def process(self):
        self.date_process()
        self.time_process()
        return self.df

