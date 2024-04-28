import os
import sys
import torch
import argparse
# =========================================================================================================
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
from model import Wav2vec_Pick
from utils import parse_arguments
import logging
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

cwb_path = "/mnt/nas5/johnn9/dataset/cwbsn/"
tsm_path = "/mnt/nas5/johnn9/dataset/tsmip/"
noi_path = "/mnt/nas5/johnn9/dataset/cwbsn_noise/"
mod_path = "/mnt/nas3/johnn9/checkpoint/"

cwb = sbd.WaveformDataset(cwb_path,sampling_rate=100)
c_train, _, _ = cwb.train_dev_test()
print(len(c_train))


mask = cwb.metadata["trace_completeness"] == 4
cwb.filter(mask)
c_train, _, _ = cwb.train_dev_test()
print(len(c_train))

p_mask = cwb.metadata["trace_p_arrival_sample"].notna()
s_mask = cwb.metadata["trace_s_arrival_sample"].notna()
cwb.filter(p_mask)
cwb.filter(s_mask)
c_train, _, _ = cwb.train_dev_test()

print(len(c_train))


sys.exit()

mask = cwb.metadata["trace_completeness"] == 1
cwb.filter(mask)
c_train, _, _ = cwb.train_dev_test()

print(type(c_train))
print(len(c_train))

p_mask = cwb.metadata["trace_p_arrival_sample"].notna()
s_mask = cwb.metadata["trace_s_arrival_sample"].notna()
cwb.filter(p_mask)
cwb.filter(s_mask)
c_train, _, _ = cwb.train_dev_test()

print(len(c_train))

sys.exit()

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}
p_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
}
s_dict = {
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",   
}

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=1000, windowlen=6000, selection="first", strategy="pad"),
    sbg.RandomWindow(windowlen=6000, strategy="pad",low=750,high=5000),
    # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
    sbg.ChangeDtype(np.float32),
    # sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
    sbg.DetectionLabeller(p_phases=p_dict, s_phases=s_dict),
]
dev_gene = sbg.GenericGenerator(t_dev)
dev_gene.add_augmentations(augmentations)

dev_loader = DataLoader(dev_gene,batch_size=100, shuffle=False, num_workers=4)
progre = tqdm(enumerate(dev_loader),total=len(dev_loader),ncols=80)
for batch_id, batch in progre:
    # General 
    y = batch['y']
    print(y.shape)

    break

y = y[:,0,:]



plt.figure(figsize=(10, 3))  # 设置图形的大小

# 遍历tensor的每一行，并绘制每一行
for i in range(y.shape[0]):
    plt.plot(y[i, :], linewidth=1, alpha=0.5)  # alpha设置透明度，使图更容易看

plt.title('Visualization of 100-dimensional Tensor')
plt.xlabel('Dimension')
plt.ylabel('Value')
savepath = './test.png'
plt.savefig(savepath)
plt.close('all')

print(y.shape)
rows_check = torch.any(torch.all(y == 0, dim=-1))

exists_zero_tensor = rows_check
print(exists_zero_tensor)