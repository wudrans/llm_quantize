##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   quantize_batch.py
@Time    :   2025/10/29 19:37:25
@Author  :   wlj 
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from utils.file_utils import get_subfolder, get_exclude_list
from quantize.quan_llmcompressor import quantize_llmcompressor
from utils.log import logger
from utils.yml_utils import yaml_load

quantize_config_path = "./configs/quantize_config.yml"


if __name__ == '__main__':
    quantize_config = yaml_load(quantize_config_path)
   
    DATASET_PATH = quantize_config['dataset_path']
    DATASET_SPLIT = "train_sft"
    dataset_path = os.path.join(DATASET_PATH, f"{DATASET_SPLIT}-*.parquet")

    saved_root = quantize_config['saved_root']

    # get all models
    pretrained_path = quantize_config['pretrained_path']
    model_list = get_subfolder(pretrained_path)
    exclude_list = ['Qwen3-30B', "FP8"]
    model_list = get_exclude_list(model_list, exclude_list, exclude_flag=True)
    # model_list = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-30B-A3B-Instruct-2507', 'Qwen3-30B-A3B-Instruct-2507-FP8', 
    #               'Qwen3-30B-A3B-Thinking-2507', 'Qwen3-30B-A3B-Thinking-2507-FP8', 
    #               'Qwen3-4B-Instruct-2507', 'Qwen3-4B-Instruct-2507-FP8', 
    #               'Qwen3-4B-Thinking-2507', 'Qwen3-4B-Thinking-2507-FP8']
    print(model_list)
    
    # model_list = ['Qwen3-4B-Thinking-2507']

    for model_name in model_list:
        model_path = os.path.join(pretrained_path, model_name)
        if not os.path.exists(model_path):
            logger.error("%s not exist" % model_path)
            continue
        logger.debug("========== %s =========="% model_path)
        
        saved_path = os.path.join(saved_root, os.path.basename(model_path))
        os.makedirs(saved_path, exist_ok=True)

        quantize_llmcompressor(model_path, dataset_path, saved_path, dataset_format="parquet")
    