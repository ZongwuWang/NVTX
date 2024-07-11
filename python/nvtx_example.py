import time
import nvtx
import torch

import os
from functools import wraps

def kill_dcgm(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        os.system('kill `pgrep -f dcgm`')
        return func(*args, **kwargs)
    return wrapper

def sleep_for(i):
    time.sleep(i)
    
# @kill_dcgm
# @nvtx.annotate("my_gemm", color="blue")
def my_gemm(a, b):
	return torch.mm(a, b)



# @kill_dcgm
# @nvtx.annotate("my_func", color="red")
def my_func():
    time.sleep(1)

with nvtx.annotate("for_loop", color="green"):
    a = torch.randn(1000, 1000).to('cuda')
    b = torch.randn(1000, 1000).to('cuda')
    for i in range(10):
        sleep_for(5)
        my_gemm(a, b)
        # my_func()


# 测试命令： ncu --set roofline --export profile_%i.ncu-rep python3 nvtx_example.py