# -*- coding:utf-8 -*-
# @Time    : 8/14/23 12:42 PM
# @Author  : wlj

import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.log import logger
import asyncio 
import chardet
import json

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPG"]
VIDEO_EXT = ['.mp4', '.avi', '.MOV']
TXT_EXT = [".txt", ".log"]


def write_file_list(path_list, saved_txt_path, shuffle=False, seed=123):
    if shuffle:
        count = len(path_list)
        s = [x for x in range(0, count)]
        random.seed(seed)
        random.shuffle(s)  # random

    fid = open(saved_txt_path, "w")
    for i,path in enumerate(path_list):
        if shuffle:
            path = path_list[s[i]].strip()
        else:
            if isinstance(path, str):
                path = path.strip()
            elif isinstance(path, int):
                path = str(path)
           
        fid.writelines(path + '\n')
    fid.close()
    return



def get_file_list(path, EXT, saved=False,saved_txt_path=None, abspath=False, 
                  shuffle=False, seed=123,  printable=True):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in EXT:
                if abspath:
                    image_names.append(apath)
                else:
                    image_names.append(filename)
    image_names.sort()
    if printable:
        logger.info("%s has object = %d"%(path, len(image_names)))
        
    if saved:
        if saved_txt_path is None:
            saved_txt_path = './temp.txt'
        write_file_list(image_names, saved_txt_path, shuffle, seed)
    return image_names


def get_subfolder(path):
    item_list = []
    for item in os.listdir(path):
        tmp = os.path.join(path, item)
        if os.path.isdir(tmp):
            item_list.append(item)
    item_list.sort()
    return item_list

    
'''
exclude_flag=True: 排除掉含有关键词的路径
exclude_flag=False: 从列表中找出包含关键词的路径
exclude_list = [fire14', 'fire15', 'fire16', 'Non-Fire']
'''
def get_exclude_list(path_list, exclude_list, saved=False,saved_txt_path=None, shuffle=False, seed=123, exclude_flag=True):
    new_path_list = []
    for path in path_list:
        flag = False
        for string in exclude_list:
            if string in path:
                flag = True
                break
        if exclude_flag:  #排除掉含有关键词的路径
            if flag:
                # logger.warning("exclude: %s" % path)
                continue
            else:
                new_path_list.append(path)
        else:   #从列表中找出包含关键词的路径
            if flag:
                new_path_list.append(path)
            else:
                continue
    new_path_list.sort()
    # print("after exclude has images = %d" % (len(new_path_list)))
    if saved:
        if saved_txt_path is None:
            saved_txt_path = './temp.txt'
        write_file_list(new_path_list, saved_txt_path, shuffle, seed)
    return new_path_list


def detect_encoding(logpath):
    # 先检测编码, 只读前 N KB 提高性能 & 准确性
    chuck_size = 1024 *10
    with open(logpath, 'rb') as f:
        raw_data = f.read(chuck_size)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']

    tmp = os.path.basename(logpath)
    show_path = tmp.split("_", 1)[1]  # 文件名的特点
    logger.debug(f"检测到编码: {encoding}, 置信度: {confidence:.2f}, file: {show_path}")

    # 如果检测到 UTF-8/GBK 等，优先尝试；否则 fallback 到更宽容的编码
    candidate_encodings = []
    if encoding is not None and confidence > 0.6:
        candidate_encodings.append(encoding)

    # 添加常见编码, 最后加入 latin1（它永远不会解码失败！）
    for enc in ['utf-8', 'gbk', 'gb2312', 'cp936', 'ascii', 'latin1']:
        if enc != encoding:
            candidate_encodings.append(enc)
    
    return candidate_encodings

'''
line_by_line: True or False
logpath: 日志文件路径
def read_file_with_config(line_by_line, logpath):
    if line_by_line:
        with open(logpath, "r") as fid:
            logdata = fid.readlines()
        # logdata[-1] = logdata[-1].strip() # 统一处理换行符
    else:
        with open(logpath, "r") as fid:
            logdata = fid.read()
    return logdata

'''
def read_file_with_config(line_by_line, logpath):
    
    candidate_encodings = detect_encoding(logpath)
    sucess_enc = None
    logdata = None

    for enc in candidate_encodings:
        try:
            # 使用 errors='replace' 避免崩溃
            with open(logpath, "r", encoding=enc, errors='replace') as fid:
                if line_by_line:
                    logdata = fid.readlines()
                # logdata[-1] = logdata[-1].strip() # 统一处理换行符
                else:
                    logdata = fid.read()
            sucess_enc = enc
            logger.debug(f"成功用编码 '{enc}' 读取文件")
            break  # 成功就退出

        except Exception as e:
            logger.warning(f"用编码 '{enc}' 读取失败: {e}")
    
    if logdata is None:
        logger.error(f"无法用任何已知编码读取文件: {logpath}")
        raise ValueError(f"无法用任何已知编码读取文件: {logpath}")
    if sucess_enc == 'latin1':
        logger.warning(f"使用 latin1 读取文件，内容可能有乱码（如 ）")

    return logdata

'''
分块读
'''
async def read_file_with_config_by_block(line_by_line, logpath):
    # 先检测编码
    candidate_encodings = detect_encoding(logpath)
    sucess_enc = None
    logdata = None

    for enc in candidate_encodings:
        try:
            if line_by_line:
                with open(logpath, "r", encoding=enc, errors='replace') as fid:
                    logdata = []
                    for k,line in enumerate(fid):
                        logdata.append(line)
                        if k % 100000 == 0:
                            await asyncio.sleep(0) #让出控制权给事件循环，允许其他就绪的协程有机会运行。
            else:
                check_size = 1024*1024*5 # 1MB
                with open(logpath, "r", encoding=enc, errors='replace') as fid:
                    logdata = []
                    while True:
                        chunk = fid.read(check_size)
                        if not chunk:
                            break
                        logdata.append(chunk)
                        await asyncio.sleep(0)

                logdata = ''.join(logdata)
                
            sucess_enc = enc
            logger.debug(f"成功用编码 '{enc}' 读取文件")
            break  # 成功就退出

        except Exception as e:
            logger.debug(f"用编码 '{enc}' 读取失败: {e}")


    if logdata is None:
        logger.error(f"无法用任何已知编码读取文件: {logpath}")
        raise ValueError(f"无法用任何已知编码读取文件: {logpath}")
    if sucess_enc == 'latin1':
        logger.warning(f"使用 latin1 读取文件，内容可能有乱码（如 ）")

    return logdata

'''
line_by_line: True or False
logpath: 日志文件路径
该函数将检查文件的大小，并根据实际情况处理，主要目的是考虑当文件较大，读取较慢时有信息反馈出去

'''
async def read_file_with_config_async(line_by_line, logpath):
    total_size = os.path.getsize(logpath) # bytes
    size_MB = total_size/1024/1024
    
    size_str = f"{total_size/1024:.2f} KB" if total_size < 1024*1024 else f"{total_size/1024/1024:.2f} MB"
    # print(f"read_file_with_config_async, file size: {size_str}")

    if size_MB < 200:  # 小于100MB的文件，直接读取
        logdata = read_file_with_config(line_by_line, logpath)
    else:
        # 分块读
        logdata = await read_file_with_config_by_block(line_by_line, logpath)

    return logdata, size_str



if __name__ == '__main__':
    import time
    # image_root_dir_host = "/data2/Detection_no_annotations/video/download"
    # saved_txt_path = "/data8T/Automatic_Test/images_list/todo.txt"
    # get_file_list(image_root_dir_host, VIDEO_EXT, saved=True,saved_txt_path=saved_txt_path, 
    #               abspath=False)
    
    logpath = "/data8T/log/HDMI二级休眠唤醒后上报热拔插事件-v0819-EVB1-83#.log"
    # read_file_with_config_async(False, logpath)

    # start = time.time()
    # logdata = read_file_with_config_by_block(True, logpath)
    # end = time.time()
    # elasped = end - start
    # print(f"block: read line by line cost {elasped:.2f} seconds")

    # start = time.time()
    # logdata = read_file_with_config_by_block(False, logpath)
    # end = time.time()
    # elasped = end - start
    # print(f"block: read line cost {elasped:.2f} seconds")


    start = time.time()
    logdata = read_file_with_config(True, logpath)
    end = time.time()
    elasped = end - start
    print(f"read line by line cost {elasped:.2f} seconds")

    # start = time.time()
    # logdata = read_file_with_config(False, logpath)
    # end = time.time()
    # elasped = end - start
    # print(f"read line cost {elasped:.2f} seconds")

def read_json(json_path):
    assert os.path.exists(json_path), f"json文件不存在: {json_path}"
    with open(json_path, "r", encoding="utf-8") as fid:
        data = json.load(fid)
    return data