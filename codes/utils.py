'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-09 18:08:41
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-09 18:09:03
FilePath: /LS-PLL-Reproduction/codes/utils.py
Description: Utils used not related to the project
'''
import os
import argparse

def validate_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"路径 '{path}' 不存在")
    return path

def validate_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' 不是有效文件夹")
    return path