# -*- coding: utf-8 -*-
# Chanson
'''
模型预测得分格式化,用户可定制开发
当某模型需要将得分进行格式化时，比如 score=.012345，格式化后为 score=S1
'''
import numpy  as np


def score_format(score,subtype,last=-1):
    t = 0.0
    if isinstance(score,np.ndarray):
        t = score[0][last]
    elif len(score)>=2:
        t = score[last]
    else:
        t = score
    # very important
    return float(t)


'''example-1

def convert_score(score):
    return score*100

def score_format(score,subtype,last=-1):
    if subtype=='ios':
        return convert_score(score[0][last])
    else:
        return score[0][last]

'''


'''example-2

def convert_score(score):
    return score*100

def score_format(score,subtype,last=-1):
    return {'predict_score':score[0][last],'format_score':convert_score(score[0][last])}

'''


'''example-3

def score_format(score,subtype,last=-1):
    return {'predict_prob_0':score[0][0],'predict_prob_1':score[0][1]}

'''
    
