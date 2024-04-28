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
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Wav2vec_Pick
from utils import parse_arguments,get_dataset
import logging
import torch.nn.functional as F
import pandas as pd
logging.getLogger().setLevel(logging.WARNING)
# =========================================================================================================
# Parameter init
args = parse_arguments()
model_name = args.model_name
print(model_name)
ptime = 500
window = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parl = 'y'  # y,n
# =========================================================================================================
mod_path = "/mnt/nas3/johnn9/checkpoint/"
model_path = mod_path + model_name
loadmodel = model_path + '/' + 'last_checkpoint.pt' 
image_path = '/mnt/nas3/johnn9/test/4_3_image/'
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
# _,_,test = get_dataset(args)

# df = pd.read_csv('/mnt/nas5/johnn9/CWBSN_seisbench/metadata.csv')
# unique_distances = df['source_magnitude'].unique()
# sorted_distances = sorted(unique_distances)
# print(sorted_distances)
# sys.exit()

data_4_3_path = "/mnt/nas5/johnn9/CWBSN_seisbench/"
test = sbd.WaveformDataset(data_4_3_path, sampling_rate=100)
mask = test.metadata['station_code'] == 'HWA'
test.filter(mask)
magmask = test.metadata['source_magnitude'] == 7.18
test.filter(magmask)
chamask = test.metadata['trace_channel'] == 'HL'
test.filter(chamask)
print(len(test))

end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")
# =========================================================================================================
# Init
def image_save(batch,x,y,savepath,num,batch_num):
    waveform1 = batch['X'][batch_num,0]
    waveform2 = batch['X'][batch_num,1]
    waveform3 = batch['X'][batch_num,2]
    p_predict = x[batch_num,0] 
    p_label = y[batch_num,0]
    waveform1 = waveform1.detach().numpy()
    waveform2 = waveform2.detach().numpy()
    waveform3 = waveform3.detach().numpy()
    p_predict = p_predict.detach().cpu().numpy()
    p_label = p_label.detach().cpu().numpy()
    # 绘制波形数据
    plt.figure(figsize=(10, 15))
    # 绘制波形数据
    plt.subplot(511)  # 第一行的第一个子图
    plt.plot(waveform1)
    plt.title('Waveform 1')
    plt.subplot(512)  # 第一行的第二个子图
    plt.plot(waveform2)
    plt.title('Waveform 2')
    plt.subplot(513)  # 第一行的第三个子图
    plt.plot(waveform3)
    plt.title('Waveform 3')
    plt.subplot(514)  # 第一行的第四个子图
    plt.plot(p_predict)
    plt.title('P_predict')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--')
    plt.subplot(515) 
    plt.plot(p_label)
    plt.title('P_label')
    plt.ylim(0, 1)
    plt.tight_layout()
    savepath = savepath + str(num) + '.png'
    plt.savefig(savepath)
    plt.close('all')

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
if args.task == 'pick':
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
        # sbg.RandomWindow(windowlen=window, strategy="pad",low=250,high=5750),
        sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
    ]
elif args.task == 'detect':
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=1000, windowlen=6000, selection="first", strategy="pad"),
        sbg.RandomWindow(windowlen=6000, strategy="pad",low=750,high=5000),
        # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
        sbg.ChangeDtype(np.float32),
        sbg.DetectionLabeller(p_phases=p_dict, s_phases=s_dict),
    ]
test_gene = sbg.GenericGenerator(test)
test_gene.add_augmentations(augmentations)
test_loader = DataLoader(test_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Whole model build
if args.train_model == "wav2vec2":
    model = Wav2vec_Pick(
        device=device,
        decoder_type=args.decoder_type,
        checkpoint_path=args.checkpoint_path,
        args=args,
    )
elif args.train_model == "phasenet":
    model = sbm.PhaseNet(phases="PSN", norm="peak")
elif args.train_model == "eqt":
    if args.task == 'pick':
        model = sbm.EQTransformer(in_samples=3000, phases='PS')
    elif args.task == 'detect':
        model = sbm.EQTransformer(in_samples=6000, phases='PS')
if parl == 'y':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_indices = list(range(num_gpus))
    model = DataParallel(model, device_ids=gpu_indices)
model.load_state_dict(torch.load(loadmodel))
model.to(device)
model.cuda(); 
model.eval()
print("Model Complete!!!")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
w2v_params = sum(p.numel() for name, p in model.named_parameters() if 'w2v' in name)
print(f"W2v params: {w2v_params}")
decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'w2v' not in name)
print(f"Decoder params: {decoder_params}")
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Model Complete time: {elapsed_time} sec")
print("=====================================================")
# =========================================================================================================
# Testing
print("Testing start!!!")
# print("Model weights:", model.module.weights)
# weights = F.softmax(model.module.weights,dim=0)
# print("Softmax weights:", weights)
# sys.exit()


start_time = time.time()
progre = tqdm(test_loader,total = len(test_loader), ncols=80)
num = 0
for batch in progre:
    x = batch['X'].to(device)
    y = batch['y'].to(device)
    x = model(x.to(device))
    if args.train_model == 'eqt':
        x_tensor = torch.empty(2,len(y),window)
        for index, item in enumerate(x):
            x_tensor[index] = item
            if index == 1:
                break
        x = x_tensor.permute(1,0,2)
        x = x.to(device)
    image_save(batch,x,y,image_path,num=num,batch_num=0)
    num = num + 1

sys.exit()