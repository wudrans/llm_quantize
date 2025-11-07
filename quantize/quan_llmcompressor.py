##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   quantize.py
@Time    :   2025/10/29 19:38:30
@Author  :   wlj 
'''
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from quantize.utils import qwen3_preprocess

'''
# dataset_format: parquet, arrow
# '''
def quantize_llmcompressor(model_path, dataset_path, saved_path, dataset_format="parquet"):
    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 256 #256
    MAX_SEQUENCE_LENGTH = 512

    # =============== 1.Load model.===============
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # =============== 2.Load dataset ===============
    # 第一种方法是直接下载数据集到缓存中，需要连接VPN才可以下载，
    # DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    # DATASET_SPLIT = "train_sft"
    # ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    # **************************************************************
    # 第二种方法先下载到缓存中，再从指定路径加载，两种方法得到的ds已验证相同
    # 使用 streaming 模式，不会一次性加载所有数据到内存
    iterable_ds = load_dataset(dataset_format, 
                    data_files=dataset_path, streaming = True)
    # 取前256个样本并转换为常规Dataset
    data_list = []
    for i, example in enumerate(iterable_ds['train']):
        if i >= NUM_CALIBRATION_SAMPLES:
            break
        # print(example)
        data_list.append(example)
    ds = Dataset.from_list(data_list) # 转换为常规Dataset
    # print("ds", ds) 
    '''example is dict:
    {'prompt': "...", 
    'prompt_id': 'XXXX', 
    'messages': [{'content': "...", 'role': 'user'}, 
                 {'content': '....', 'role': 'assistant'}, 
                 {'content': '...', 'role': 'user'}, 
                 {'content': "...", 'role': 'assistant'}, ......]
    }

    ds:
    Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 2
    })
    prompt:提问词
    messages:是一个列表，通常包含多轮对话，每轮对话是一个字典
    如何从ds从取样本
    ds[0]代表第0个样本，是一个dict,包含的key:prompt,prompt_id, messages'''
    
    
    # =============== 3.preprocess ===============
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda example: {"text": qwen3_preprocess(example["messages"], tokenizer, model_path)})
    # Dataset({
    #     features: ['prompt', 'prompt_id', 'messages', 'text'],
    #     num_rows: 2
    # })


    # =============== 4.quantization ===============
    # quantization config
    recipe = [AWQModifier(
            # ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
            ignore=[
                "lm_head", 
                "embed_tokens",  # 添加embedding层
                "re:.*norm.*",   # 忽略所有norm层
                "re:.*mlp.gate$", 
                "re:.*mlp.shared_expert_gate$",
                "re:.*attention.*output.*",  # 考虑忽略注意力输出层
            ],
            scheme="W4A16",
            targets=["Linear"],
        ),
    ]

    oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )
    

    # =============== 5.save model ===============
    model.save_pretrained(saved_path, save_compressed=True)
    tokenizer.save_pretrained(saved_path)
    print("sucessfuly saved to %s" % saved_path)


    # =============== 6. verify ===============
 
    print("\n=============== 6. verify ===============")

    from algorithms.LLM.Qwen import ModelQwen

    input_message = [{'content': "你好，我是", 'role':'user'}]
    

    print("=================original output=====================")
    model_hf = ModelQwen(model_path)
    original_ouput_ = model_hf.inference(input_message)
    original_ouput = model_hf.post_process(original_ouput_, model_hf.input_tokens)
    print(original_ouput)
    
    print("\n=================quantize output=====================")
    model_quan = ModelQwen(saved_path)
    quan_ouput_ = model_quan.inference(input_message)
    quan_ouput = model_quan.post_process(quan_ouput_, model_quan.input_tokens)
    print(quan_ouput)


    





    