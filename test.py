import torch
import time
import numpy as np
# this ensures that the current Mac_OS version is at least 12.3+
# print(torch.backends.mps.is_available())
# # this ensures that the current PyTorch installation was built with MPS activated.
# print(torch.backends.mps.is_built())
# # To run PyTorch code on the GPU, use torch.device("mps") analogous to torch.device("cuda") on an Nvidia GPU.
# device = torch.device("mps")
#
# time1 = time.time()
# time.sleep(10)
# time2 = time.time()
# print(time2 - time1)

# 假设你有一个20*1的矩阵

# 设置要重复的值
value = [1,2,3,4]
value = np.array(value)
# 使用numpy的repeat函数来创建一个一维矩阵


print(value*100)

