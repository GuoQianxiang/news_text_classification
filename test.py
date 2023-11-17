import torch
import time
import numpy as np
import pandas as pd

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
df = pd.DataFrame(columns=['label'])
print(df)
label_list = ['label1', 'label2', 'label3', 'label4']

df['label'] = label_list

print(df)
