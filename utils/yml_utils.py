##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yml_utils.py
@Time    :   2025/07/22 18:37:28
@Author  :   wlj 
'''

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import yaml
from utils.log import logger

def yaml_load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.error("%s not exists" % path)
        exit()
    return config



def write_yaml(yaml_path, config_info):
    with open(yaml_path, "w", encoding="utf-8") as fid:
        yaml.safe_dump(config_info, fid, sort_keys=False, 
                       default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    # 测试用例
    path = "./resources/log关键词.yml"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    
    for k, key in enumerate(config.keys()):
        print(k, key)
