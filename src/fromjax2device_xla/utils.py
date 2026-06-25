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
    paths = {} 
    path = Path('/tmp/xla_dump')
    for item in path.iterdir():
        if item.is_file():
            filename = item.name
            if module_name in filename:
                if filename.endswith('before_optimizations.txt'):
                    paths['before_optimizations'] = str(item)
                if filename.endswith('after_optimizations.txt'):
                    paths['after_optimizations'] = str(item)
                if 'after_pipeline-start' in filename:
                    paths['before_gemm_rewriter'] = str(item)
                if 'gemm-rewriter' in filename:
                    paths['gemm_rewriter'] = str(item)
                if filename.endswith('thunk_sequence.txt'):
                    paths['thunk_sequence'] = str(item)
                if filename.endswith('.ptx'):
                    paths['ptx'] = str(item)
                if filename.endswith('buffer-assignment.txt'):
                    paths['buffer_assignment'] = str(item)
        return paths


def clean_dump():
    jax.clear_caches()
    path = Path('/tmp/xla_dump')
    if path.exists():
        shutil.rmtree(path)
def clean_hlo(text):
    # Remove metadata={...} (including everything inside the brackets)
    text = re.sub(r',\s*metadata=\{[^}]*\}', '', text)

    # Remove the layout brackets like {1,0} or {0,1} right after dimensions
    text = re.sub(r'\{\d+(,\d+)*\}', '', text)

    return text
def validate_dump_path(path):
    if not path.startswith('/tmp/xla_dump/'):
        path = '/tmp/xla_dump/' + path
    return path
def print_file(path, is_hlo = True):
    path = validate_dump_path(path)
    with open(path, 'r') as file:
        hlo = file.read()
    if is_hlo:
        hlo = clean_hlo(hlo)
        print(hlo)

def compare_hlo(file1_path, file2_path,width = 80):

    file1_path = validate_dump_path(file1_path)
    file2_path = validate_dump_path(file2_path)

    with open(file1_path, 'r') as f:
        f1_lines = map(clean_hlo, f.read().splitlines())
    with open(file2_path, 'r') as f:
        f2_lines = map(clean_hlo, f.read().splitlines())

    
def hlo_to_graph(hlo_text):
    module = hlo_module_from_text(hlo_text)
    dot_graph = hlo_module_to_dot_graph(module)
    source = graphviz.Source(dot_graph)
    display(source)
    

###################Xprof Helpers###########################
import glob
import shutil
import os

def download_xprof():
    # Define the base directory
    base_path = '/tmp/tensorboard/plugins/profile/'

    # Check if directory exists first
    if not os.path.exists(base_path):
        print("Base path not found.")
        return

    # Get all subdirectories
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]

    if not subdirs:
        print("No folders found in the specified path.")
    else:
        # Find the newest directory
        latest_folder = max(subdirs, key=os.path.getmtime)
        print(f"Targeting newest folder: {latest_folder}")

        # Prepare for zipping
        # root_dir is '/tmp' so the zip includes the 'tensorboard' folder
        # base_dir is the relative path (e.g., 'tensorboard/plugins/profile/2026...')
        root_dir = '/tmp'
        base_dir = os.path.relpath(latest_folder, start=root_dir)
        zip_filename = os.path.basename(latest_folder) # Uses the timestamp as the filename

        print(f"Zipping '{base_dir}'...")

        # Create the zip archive
        shutil.make_archive(zip_filename, 'zip', root_dir=root_dir, base_dir=base_dir)

        # Download the zip
        print(f"Downloading {zip_filename}.zip...")
        files.download(f'{zip_filename}.zip')
   
def profile(function, *args):
    jax.clear_caches()
    log_dir = "/tmp/tensorboard/"
    with jax.profiler.trace(log_dir):
        result = jax.jit(function)(*args)
    result.block_until_ready()
