import os
# JAX 是一个 Python 数值计算库。你可以把它理解成“像 NumPy 一样写数组计算，
# 但可以自动求导、自动向量化，并且能把代码编译到 CPU/GPU/TPU 上跑
#
# HLO 是 XLA 编译器里面的一种中间表示，全称通常是 High Level Optimizer 或 High Level Operations。
# 你可以把它理解成：JAX 代码被编译前，会先被翻译成一种更接近编译器能理解的计算图语言。
#
# Jaxpr 是 JAX 自己的一种“中间表示”。
# 你可以把它理解成：
# JAX 把你的 Python 函数追踪一遍之后，得到的一份简化计算说明书。
#

#设置环境变量, 把中间结果dump出来
#在import jax之前设置, 否则不会生效
os.environ["XLA_FLAGS"] = (
    #文件保存地址
    '--xla_dump_to=/tmp/xla_dump'
    #启用cublaslt, xla 使用nvida的线性代数标准计算库
    '--xla_gpu_enable_cublaslt=true' \
    # 让HLO dump为文本格式, 方便查看
    '--xla_dump_hlo_as_text' \
    # 只对指定的HLO pass进行dump
    '--xla_dump_hlo_pass_re=.* '
)

import jax
import jax.numpy as jnp

print("jax version:", jax.__version__)
print("jax available devices:", jax.devices())
print("jax default device:", jax.default_backend())

#XLA utils 
import intertools
import textwrap
import re
from pathlib import Path
import shutil
import graphviz
from IPython.display import display



import jaxlib.xla_client as xla_client

hlo_module_from_text = xla_client.xla.hlo_module_from_text
hlo_module_to_dot_graph = xla_client.xla.hlo_module_to_dot_graph

def get_dump_paths(module_name):
    pass
def clean_dump():
    pass
def clean_hlo(text):
    pass
def validate_dump_path(path):
    pass
def print_file(path, is_hlo = True):
    pass
def compare_hlo(file1_path, file2_path,width = 80):
    pass
    
def hlo_to_graph(hlo_text):
    pass
    

###################Xprof Helpers###########################
import glob
import shutil
import os

def download_xprof():
    pass    
def profile(function, *args):
    pass
