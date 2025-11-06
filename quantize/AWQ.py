##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   awq.py
@Time    :   2025/10/29 10:40:55
@Author  :   wlj 
'''

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os


if __name__ == '__main__':
    model_path = '/data/wlj/pretrained/Qwen/Qwen3-0.6B'
    quant_path = '/data/wlj/pretrained/Qwen_AWQ/Qwen3-0.6B'
    os.makedirs(quant_path, exist_ok=True)

    # To use Marlin, you must specify zero point as False and version as Marlin.
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')
