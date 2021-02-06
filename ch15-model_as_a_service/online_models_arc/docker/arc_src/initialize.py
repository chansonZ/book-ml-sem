# -*- coding: utf-8 -*-
# Chanson
import yaml
import os 
import re
from utils import get_model,get_conf_dict,exit_with_mess

# key与conf文件匹配
_m11    = 'server_conf'
_m12    = 'sub_models'
_m13    = 'other_define'
_m14    = 'data_process'

_m11_m  = 'model_name'
_m11_log = 'key'

_m12_m  = 'model_file'
# 这两个特殊-内外部都可以有
_m12_d  = 'data_dict'
_m12_o  = 'data_order'


class EnvInit():
    def __init__(self,user_conf_path,resource_path,default_conf_path='_conf.yml'):

        if not os.path.isfile(default_conf_path):
            exit_with_mess("default_conf_path:{} is not a valid file".format(default_conf_path))
        
        self.models={}
        self.data_dict={}
        self.data_order={}
        self.server_conf={}
        self.data_process={}

        t = get_conf_dict(default_conf_path)

        if os.path.isfile(user_conf_path):
            u = get_conf_dict(user_conf_path)
        else:
            exit_with_mess('There is no user configure file:conf.yml in {}'.format(user_conf_path))

        # 服务配置
        self.server_conf = t[_m11]
        self.server_conf.update({_m11_m:u[_m11_m]})
        if _m11_log in u.keys() and u[_m11_log] is not None:
            self.server_conf.update({_m11_log:u[_m11_log]})

        if _m14 in u.keys() and u[_m14] is not None:
            self.data_process = u[_m14]
 
        def _update(target_dict,kvs,key1,key2):
            # 当sub_models的下级字段中没有 data_order 或 data_dict 时，将从全局（外层）中继承，如果都没有，则错误退出
            if key2 not in kvs[key1].keys():
                # 看是否在全局：u
                if key2 not in u.keys():
                    exit_with_mess("ERROR:'{}' not in '{}' sub field,check conf.yml".format(key2,key1))
                elif u[key2] is None:
                    exit_with_mess("ERROR:'{}' not in '{}' sub field,check conf.yml".format(key2,key1))
                target_dict.update({key1: u[key2] })
            else:
                target_dict.update({key1: kvs[key1][key2] })
            return target_dict
        
        #展平，加载二进制模型
        models = u[_m12]
        for mm in models.keys():   
            if models[mm][_m12_m] is None:
                model_path = None
            else:
                model_path = os.path.join(resource_path,models[mm][_m12_m])
                if not os.path.exists(model_path):
                    exit_with_mess("ERROR:{} is not a file! Check conf.yml:'model_file'".format(model_path))
            self.models.update({mm: get_model( model_path ) })
            self.data_dict = _update(self.data_dict,models,mm,_m12_d)
            self.data_order = _update(self.data_order,models,mm,_m12_o)

        