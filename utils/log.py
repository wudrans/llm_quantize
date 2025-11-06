# -*- coding:utf-8 -*-
# @Time    : 7/3/23 5:53 PM
# @Author  : wlj

import os
import yaml
import logging.config

def setup_logging(yaml_path='utils/log.yaml'):
    if yaml_path is None:
        # 避免相对路径带来的问题
        yaml_path = os.path.join(os.path.dirname(__file__), "log.yml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)

    else:
        print("%s not exists, log print on console" % yaml_path)
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')


'''
将logger设置为全局变量，整个项目使用同一个logger去打印
'''
setup_logging(yaml_path='utils/log.yml')
logger = logging.getLogger('detailed') #实例化记录器