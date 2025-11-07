##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   apply_LLM.py
@Time    :   2025/07/11 15:02:36
@Author  :   wlj 
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

# 早期的Qwen3版本
previous_Qwen3 = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

'''
model_path: 模型路径,可以是本地路径或Hugging Face模型名称

torch_dtype="auto",将根据检查点的原始精度和设备支持的精度自动确定要使用的数据类型。
对于现代设备，确定的精度将是 bfloat16, 默认数据类型为 float32, 这将占用两倍的内存并且计算速度较慢。
device_map="auto", #多GPU情况,自动将模型参数加载到多个设备上,它依赖于 accelerate 包           
'''
class ModelQwen:
    def __init__(self, model_path):

        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          dtype="auto",
                                                          device_map="auto")
        
        # 加载模型对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 

        self.model_path = model_path
        self.max_new_tokens = 100
        self.use_cache = True
        self.do_sample = False
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.num_beams = 1
        self.repetition_penalty = 1.1
        self.thinking = False
        self.input_tokens = 0
        self.input_chars = 0
        self.computed_dtype = self.model.dtype
        self.max_model_len = 4096
    

    def inference(self, messages_list=None, streamer=None):
        # 按消息（message）截断（用于 chat 模板）
        # messages_list = truncated_template_inputs(self.tokenizer, messages_list, int(self.max_model_len*0.8))
        # print("messages_list", messages_list)

        basename = os.path.basename(self.model_path)
        if basename in previous_Qwen3:
            # Qwen3-1.7B模型使用enable_thinking, Default is True.
            text = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False, # 是否要将文本转换为token ID序列
                add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
                enable_thinking=self.thinking, # Switches between thinking and non-thinking modes. Default is True.
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False, # 是否要将文本转换为token ID序列
                add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
            )

        # print("Input text:", text)
        '''
        Input text: <|im_start|>system
        使用中文回答<|im_end|>
        <|im_start|>user
        请写一篇文章：我的妈妈，不少于1000字<|im_end|>
        <|im_start|>assistant
        <think>

        </think>
        '''
        #对文本进行分词,return_tensors="pt"返回 PyTorch 张量
        # truncation=True 自动截断到最大长度
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # print('model_inputs', model_inputs)
        '''
        model_inputs {'input_ids': tensor([[151644,   8948,    198,  37029, 104811, 102104, 151645,    198, 151644,
            872,    198,  14880,  61443, 116562,   5122,  97611, 101935, 151645,
            198, 151644,  77091,    198, 151667,    271, 151668,    271]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1]], device='cuda:0')}
         '''
        # prompt的实际token数
        input_length = len(model_inputs.input_ids[0])
        self.input_tokens = input_length
        self.input_chars = len(text)

        '''
        使用模型进行推理, 加no_grad(),显存占用更低,跳过梯度相关数据的保存
        当do_sample=False时,使用贪婪搜索，结果确定,适合于代码生成/事实问答;
        当do_sample=False时,以下参数会被忽略:temperature,top_p,top_k
        do_sample=True + temperature→0	接近贪婪搜索（但仍是采样过程）	弱随机性需求
        do_sample=True + temperature>1.0	放大低概率token的权重	创意写作/头脑风暴
        '''
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=self.max_new_tokens,  # 设置生成的最大token数
                do_sample=self.do_sample, # 启用采样,允许生成多样化的输出; 
                num_beams=self.num_beams, # 启用采样,允许生成多样化的输出;

                # past_key_values=None, # 不使用过去的key/value缓存,通常用于加速多轮生成
                use_cache=self.use_cache,  # 使用缓存以加速推理
                temperature=self.temperature, # 控制输出的随机性,值越高输出越随机
                top_p=self.top_p, # 使用 nucleus sampling, 只考虑概率累积到 top_p 的令牌
                top_k=self.top_k,
                repetition_penalty = self.repetition_penalty, #默认无重复惩罚
                streamer=streamer,  # 使用流输出对象处理生成的文本流
                )
            
        return generated_ids
    
            
    def post_process(self, generated_ids, input_length):
        # 从生成结果中提取新生成的 token ID（去掉输入部分）
        output_ids = generated_ids[0][input_length:].tolist() 

        if self.thinking:
            # 解析思考内容和实际内容
            try:
                # 尝试找到 </think> 标记（ID 151668）的位置
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # print("Index of thinking content end:", index)

            '''
            使用分词器将 token ID 解码为文本
            skip_special_tokens=True: 跳过特殊 token
            strip("\n"): 去除首尾换行符
            '''

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            thinking_content = ''
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
        return [thinking_content, content]
