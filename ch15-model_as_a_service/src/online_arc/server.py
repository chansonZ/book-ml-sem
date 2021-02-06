# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第15章 模型上线

from flask import Flask,request
from flask_restful import Api,Resource
import os 

from initialize import EnvInit
from utils import *
import logging
import sys


# iris模型上线示例
_user_src_path = '../iris_online_example/src'
_user_resource_path = '../iris_online_example/resource'
_user_log_path = '../iris_online_example/log'

_user_dataprocess=os.path.join(_user_src_path,'dataprocess.py')
_user_scorefromat=os.path.join(_user_src_path,'scorefromat.py')
_user_conf_path = os.path.join(_user_src_path,'conf.yml')
sys.path.append(_user_src_path)

if os.path.isfile(_user_dataprocess):
    from dataprocess import data_process
else:
    from _dataprocess import data_process

if os.path.isfile(_user_scorefromat):
    from scorefromat import  score_format
else:
    from _scorefromat import score_format

# 单条数据格式
JSON_KEY_LIST=['submodel','key','data']
PACKAGE_KEY='package'
'''
{
    "subtype":"child1",
    "key":"122345688999",
    "data":{xxxx}
}
'''
################################################################################
from enum import Enum,unique
@unique
class ErrCode(Enum):
    SUCCESS=0
    NO_PARA=101
    JSON_ERR=102
    DATA_NOT_MATCH=103
    NOT_SUPPORT=201
    USER_ERR1=301
    USER_ERR2=302
    SERVER_ERR=401
################################################################################
#全局变量
app = Flask(__name__)
api = Api(app)
g_models = EnvInit(user_conf_path=_user_conf_path,resource_path=_user_resource_path,default_conf_path='_conf.yml')
server_url = '/models/{}/'.format(g_models.server_conf['model_name'])
g_hint= '\n Please visit: http://ip:{}{} \n '.format(g_models.server_conf['server_port'],server_url)
################################################################################

@app.before_request
def pre_request_logging():
    # 会很大哦
    app.logger.info(request.url)

@app.before_first_request
def set_logging():
    if not os.path.isdir(_user_log_path):
        os.makedirs(_user_log_path)
    logfile = os.path.join(_user_log_path, '{}.log'.format(g_models.server_conf['model_name']))
    app.logger.addHandler(create_rotating_log_handler(logfile,g_models.server_conf['log_level']))
    app.logger.info(g_models.server_conf)
    app.logger.info(g_hint)

def check_package_valid(web_args):
    return validate_data(JSON_KEY_LIST,web_args.keys())


def check_single_args(web_args):
    if web_args is None or len(web_args)==0:
        return ErrCode.NO_PARA.value,'nothing get from http request!'#无参

    code,msg = check_package_valid(web_args)
    if code != ErrCode.SUCCESS.value:
        return ErrCode.JSON_ERR.value,msg # JSON格式字段不匹配

    subtype     = get_dict_value(web_args,JSON_KEY_LIST[0])
    data        = get_dict_value(web_args,JSON_KEY_LIST[2])

    if subtype is None:
        return ErrCode.JSON_ERR.value,"request should carry 'subtype' field"

    if subtype not in g_models.models.keys():
        return ErrCode.NOT_SUPPORT.value,"model subtype '{}' not support in :{}".format(subtype,g_models.models.keys())

    if g_models.server_conf['validate']:
        print('in validate:',g_models.data_dict[subtype])
        need_keys = list(g_models.data_dict[subtype].keys())
    ret, msg = validate_data(need_keys,data.keys())
    if ErrCode.SUCCESS.value != ret:
        return ErrCode.DATA_NOT_MATCH.value,msg
    return ErrCode.SUCCESS.value,''


def format_return(key,sub_type,code=-1,msg='',result=''):
    return {"key":str(key),"subtype":sub_type,"result":{"code":code,"msg":msg,"score":result}}

def check_orgin_data(data,subtype):
    return validate_data(g_models.data_dict[subtype].keys(),data.keys())
    
def check_processed_data(data,subtype):
    return validate_data(g_models.data_order[subtype],data.keys())

def predict_single(args):
    '''返回dict格式，对应返回的 JSON:
    return:
        {"key":str(key),"subtype":sub_type,"return":{"code":code,"msg":msg,"result":result}}
    '''
    subtype = get_dict_value(args,JSON_KEY_LIST[0])
    key     = get_dict_value(args,JSON_KEY_LIST[1])
    data    = get_dict_value(args,JSON_KEY_LIST[2])

    # 参数检查
    code,msg = check_single_args(args)
    if code != ErrCode.SUCCESS.value:
        return format_return(key,subtype,code,msg,result='')

    # 字典检查
    code,msg = check_orgin_data(data,subtype)
    if code != ErrCode.SUCCESS.value:
        return format_return(key,subtype,code,msg,result='')

    # 数据处理
    d = data_process(data,subtype,**g_models.data_process)
    if not isinstance(d,dict):
        msg = 'data_process() failed，it should return a dict,but return:{}'.format(d)
        return format_return(key,subtype,ErrCode.USER_ERR1.value,msg,result='') # 用户未按规范实现接口

    app.logger.debug('after data_process:{}'.format(d))

    # 特殊模型-不太考虑这种实现
    if g_models.models[subtype] is None:
        app.logger.debug('g_models.models[{}] is None'.format(subtype))
        _bad_msg = "There is no bin model file and data_process.py failed! You should return format {'code':xx,'result':yy,'msg':'success or others'}"
        if set(['code','result','msg'] != set(d.keys())):
            return format_return(key,subtype,ErrCode.USER_ERR1.value,_bad_msg,result='')# 用户未按规范实现接口
        else:
            #特征模型不格式化了，要求直接输出目标结果
            return format_return(key,subtype,d['code'],d['msg'],result=d['result'])
    else:
        code, msg = check_processed_data(d,subtype)
        if code != ErrCode.SUCCESS.value:
            return format_return(key,subtype,ErrCode.USER_ERR2.value,'After data_process():'+msg,result='')#用户实现与配置不一致
        d = data_order(d,g_models.data_order[subtype])

        #这里优先使用了 predict_proba
        try:
            result = g_models.models[subtype].predict_proba(d)
        except Exception as e:
            result = g_models.models[subtype].predict(d)    	
    
        result = score_format(result,subtype)
        return format_return(key,subtype,ErrCode.SUCCESS.value,'success',result=result)

def predict(web_args):
    if PACKAGE_KEY in web_args.keys():
        ret = [predict_single(args) for args in web_args[PACKAGE_KEY]]
        return {PACKAGE_KEY:ret}
    else:
        return predict_single(web_args)
    
class PredictHandler(Resource):
    @classmethod
    def _predict(cls,dict_data):
        try:
            ret = predict(dict_data)
        except Exception as e:
            ret = format_return("","",ErrCode.SERVER_ERR.value,msg=str(e)+":{}".format(dict_data),result='')
            app.logger.error("{}".format(ret))

        app.logger.info("[receive] {}".format(dict_data))
        app.logger.info("[send] {}".format(ret))

        return ret
        
    def post(self):
        return PredictHandler._predict(request.get_json(force=True))

# url 
api.add_resource(PredictHandler,server_url)
print(g_hint)

if __name__ == '__main__':
    app.run(
        host=str(g_models.server_conf['host']),
        port=g_models.server_conf['server_port'],
        debug=g_models.server_conf['debug'])

else:
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(gunicorn_error_logger.level)
