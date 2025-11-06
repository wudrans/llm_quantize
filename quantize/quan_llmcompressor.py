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
from llmcompressor.utils import dispatch_for_generation
import os


# 早期的Qwen3版本
previous_Qwen3 = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

# 预处理成标准格式，如<|im_start|>user... <|im_end|>
def preprocess(example, tokenizer, model_path):
    basename = os.path.basename(model_path)
    if basename in previous_Qwen3:
        # Qwen3-1.7B模型使用enable_thinking, Default is True.
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )
    else:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
        )
    result = {"text": text}
    # print(result)
    return result
    '''
    text: 
        <|im_start|>user
        请写一篇文章：我的妈妈，不少于1000字
        <|im_end|>
        <|im_start|>assistant
        ......
        <|im_end|>
        <|im_start|>user
        ......
        <|im_end|>
        ....
    '''

def model_test(model, tokenizer, input_message, max_new_tokens):
    input_ids = tokenizer(input_message, return_tensors="pt").to(model.device)
    input_length = len(input_ids.input_ids[0])

    output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output[0][input_length:])
    # print(result)
    return result

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

    # original modet test
    input_message = "你好，我是"
    max_new_tokens = 128
    original_ouput = model_test(model, tokenizer, input_message, max_new_tokens)

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
        data_list.append(example)
    ds = Dataset.from_list(data_list) # 转换为常规Dataset
    # print("ds", ds) 
    '''ds:
    Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 2
    })
    prompt:提问词
    messages:是一个列表，通常包含多轮对话，每轮对话是一个字典
    [{'content':"....", 'role':'user'}, {'content':"....", 'role':'assistant'}, ....]
    如何从ds从取样本
    ds[0]代表第0个样本，是一个dict,包含的key:prompt,prompt_id, messages'''
    
    
    # =============== 3.preprocess ===============
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda example: preprocess(example, tokenizer, model_path))
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
    
    # =============== 5. verify ===============
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer(input_message, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    
    input_length = len(input_ids.input_ids[0]) # 去掉输入信息
    print(tokenizer.decode(output[0][input_length:]))

    print("=================quantize output=====================\n\n")
    quantize_ouput = model_test(model, tokenizer, input_message, max_new_tokens)
    print(quantize_ouput)

    print("=================original output=====================\n\n")
    print(original_ouput)



    # =============== 6.save model ===============
    model.save_pretrained(saved_path, save_compressed=True)
    tokenizer.save_pretrained(saved_path)
    print("sucessfuly saved to %s" % saved_path)



    