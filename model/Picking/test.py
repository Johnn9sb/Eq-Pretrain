import os
import sys
import hydra
import torch
import argparse
# =========================================================================================================
import seisbench.data as sbd
import seisbench.generate as sbg
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import time

tsm = sbd.WaveformDataset('/work/u3601026/dataset/tsmip/',sampling_rate=100)
t_mask = tsm.metadata["trace_completeness"] == 1
tsm.filter(t_mask)
train, _, _ = tsm.train_dev_test()

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
augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
    sbg.RandomWindow(windowlen=3000, strategy="pad"),
    # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)
]

train_gene = sbg.GenericGenerator(train)
train_gene.add_augmentations(augmentations)
train_loader = DataLoader(train_gene,batch_size=100, shuffle=True, num_workers=4,pin_memory=True)

import matplotlib.pyplot as plt

for batch_id, batch in enumerate(train_loader):
    print(batch.keys())
    y = batch['y'][:,0,:]
    print(y.shape)
    rows_with_all_zeros = torch.all(y == 0, dim=1)
    print("Any row with all zeros?", torch.any(rows_with_all_zeros).item())


    plt.figure(figsize=(10,6))

    for i in range(y.shape[0]):
        plt.plot(y[i,:])
    plt.title('label')
    plt.savefig('./test.png')
    plt.close()
    break

