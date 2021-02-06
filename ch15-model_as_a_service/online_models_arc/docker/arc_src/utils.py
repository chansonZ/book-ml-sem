# -*- coding: utf-8 -*-
# Chanson
import yaml
import os 
import re
import sys
import numpy as np

def exit_with_mess(mess,exit_code=-1):
    print(mess)
    sys.exit(exit_code)

def format_yml_file(file):
    ''' simple replace'''
    src   = ':'
    to    = ': '
    bak_file = "%s.bak" % file
    with open(file, "r") as f1,open(bak_file, "w") as f2:
        for line in f1:
            if src in line:
                line = line.replace(src, to)
            f2.write(re.sub(':[ ]{2,}',to,line))
    os.remove(file)
    os.rename(bak_file, file)

def read_ymlconf(conf_file,format=True):
    if format:
        format_yml_file(conf_file)
    with open(conf_file) as infile:
        return yaml.load(infile)

def get_conf_dict(para_dict):
    _BADMESS = 'config should be a:[a file path,dict]'

    if isinstance(para_dict, str):
        if os.path.isfile(para_dict):
            return read_ymlconf(para_dict)
        else:
            exit_with_mess('{} :is not a valid path,exit -1'.format(para_dict))
    elif isinstance(para_dict, dict):
        return para_dict
    else:
        exit_with_mess(_BADMESS)


def load_pickle_modelfromfile(model_path):
    """ https://docs.python.org/2/library/pickle.html#pickle.load """
    import pickle
    with open(model_path, 'rb') as fo:
        return pickle.load(fo)

def load_joblib_model(model_path):
    from sklearn.externals import joblib
    return joblib.load(model_path)  if os.path.isfile(model_path) else None

def get_model(model,type='joblib'):
    """ Get model form RAM or pickle or joblib file """
    if model is None:
        return None
    if isinstance(model,str) and type == 'joblib':
        return load_joblib_model(model)
    elif type == 'pickle':
        return load_pickle_modelfromfile(model)
    else:
        return model

def data_format(data,data_dict):
    '''数据格式化
    data:webapi传入的dict型数据
    data_dict：为数据字典，data中的数据要求按照该格式格式化为 int float，str等

    输出：
        格式化好的dict型数据
    '''
    _f={
    'int':int,
    'string':str,
    'float':float
    }

    ret = {}
    if data_dict is None:
        return data
    for k, v in data_dict.iteritems():
        ret.update({k:_f.get(v,str)(data[k])})
    return ret

    
def data_order(data,data_order):
    '''数据按模型训练顺序排序
    data：字典型数据
    输出：list，后直接进入模型预测
    仅支持单行样本
    '''
    if isinstance(data,dict):
        return np.array([ data[aa] for aa in data_order ]).reshape(1, -1)
    else:
        return None


def validate_data(columns,args_keys):
    _bad_len='The input paramters:{} do not match the right parameters:{}'.format(args_keys,list(columns))
    if set(args_keys) != set(list(columns)):
        return -1,_bad_len
    return 0,''

def get_dict_value(args,key):
    if not isinstance(args,dict):
        return None
    return args.get(key,None)

def get_model_data(args,key='type'):
    '''
    获取特征
    '''
    if key is not None:
        for aa in args:
            aa.pop(key)
    return args

def create_rotating_log_handler(logfile,log_level='debug',maxBytes=10485760,backupCount=5):
    import logging
    from logging.handlers import RotatingFileHandler

    t = os.path.dirname(logfile)
    if not os.path.isdir(t):
            os.makedirs(t)

    fileHandler = RotatingFileHandler(logfile,maxBytes=maxBytes,backupCount=backupCount)
    
    if log_level=='debug':
        fileHandler.setLevel(logging.DEBUG)
    else:
        fileHandler.setLevel(logging.INFO)
    
    fileHandler.setFormatter(
        logging.Formatter( #[%(threadName)s] [%(pathname)s:%(lineno)d]
            '[%(asctime)s] [%(levelname)s] %(message)s',
            '%Y%m%d %H:%M:%S'))
    return fileHandler
