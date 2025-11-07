##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/11/06 18:05:01
@Author  :   wlj 
'''

import os

# 早期的Qwen3版本
previous_Qwen3 = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

# 
'''
messages_list:  [{'content': "...", 'role': 'system'}, 
                {'content': "...", 'role': 'user'}, 
                {'content': '....', 'role': 'assistant'}, 
                {'content': '...', 'role': 'user'}, 
                {'content': "...", 'role': 'assistant'}, ......]

预处理成标准格式，如<|im_start|>user... <|im_end|>
<|im_start|>assistant... <|im_end|>

'''
def qwen3_preprocess(messages_list, tokenizer, model_path):
    basename = os.path.basename(model_path)
    if basename in previous_Qwen3:
        # Qwen3-1.7B模型使用enable_thinking, Default is True.
        text = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )
    else:
        text = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
        )
   
    return text
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


if __name__ == '__main__':
    a = 0
