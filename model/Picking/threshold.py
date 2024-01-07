import os
import sys
import torch
import argparse
# =========================================================================================================
import seisbench.data as sbd
import seisbench.generate as sbg
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import time
from model import Wav2vec_Pick
from utils import parse_arguments
import logging
logging.getLogger().setLevel(logging.WARNING)

# =========================================================================================================
# Parameter init
args = parse_arguments()
test_name = 'pick_threshold'
model_name = args.model_name
print(model_name)
ptime = 500
window = 3000
parl = 'y'  # y,n
threshold = [0.1,0.2,0.3,0.4,0.5,0.6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================================================================
cwb_path = "/mnt/nas5/johnn9/dataset/cwbsn/"
tsm_path = "/mnt/nas5/johnn9/dataset/tsmip/"
noi_path = "/mnt/nas5/johnn9/dataset/cwbsn_noise/"
mod_path = "/mnt/nas3/johnn9/checkpoint/"
model_path = mod_path + model_name
threshold_path = model_path + '/' + test_name + '.txt'
loadmodel = model_path + '/' + 'best_checkpoint.pt' 
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
cwb = sbd.WaveformDataset(cwb_path,sampling_rate=100)
c_mask = cwb.metadata["trace_completeness"] > 3
cwb.filter(c_mask)
_, c_dev, _ = cwb.train_dev_test()
tsm = sbd.WaveformDataset(tsm_path,sampling_rate=100)
t_mask = tsm.metadata["trace_completeness"] == 1
tsm.filter(t_mask)
_, t_dev, _ = tsm.train_dev_test()
if args.noise_need == 'true':
    noise = sbd.WaveformDataset(noi_path,sampling_rate=100)
    _, n_dev, _ = noise.train_dev_test()
    dev = c_dev + t_dev + n_dev
elif args.noise_need == 'false':
    dev = c_dev + t_dev
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")
# =========================================================================================================
# Funtion
def label_gen(label):
    # (B,3,3000)
    label = label[:,0,:]
    label = torch.unsqueeze(label,1)
    # other = torch.ones_like(label)-label
    # label = torch.cat((label,other), dim=1)

    return label

print("Function load Complete!!!")
# =========================================================================================================
# Init
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
    sbg.RandomWindow(windowlen=window, strategy="pad"),
    # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)
]
dev_gene = sbg.GenericGenerator(dev)
dev_gene.add_augmentations(augmentations)
dev_loader = DataLoader(dev_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Wav2vec model load
model = Wav2vec_Pick(
        device=device,
        decoder_type=args.decoder_type,
        checkpoint_path='None'
)
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
print("Threshold start!!!")
start_time = time.time()
f = open(threshold_path,"w")
for thres in threshold:
    
    print("Threshold: " + str(thres) + " start!!")
    p_tp,p_tn,p_fp,p_fn = 0,0,0,0
    s_tp,s_tn,s_fp,s_fn = 0,0,0,0

    progre = tqdm(dev_loader, total=len(dev_loader), ncols=80)
    for batch in progre:
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        y = label_gen(y.to(device))
        x = model(x.to(device))

        if args.model_type == 'onlyp':
            for num in range(len(x)):
                xp = x[num,0]
                yp = y[num,0]
                    
                if torch.max(yp) >= thres and torch.max(xp) >= thres:
                    if abs(torch.argmax(yp).item() - torch.argmax(xp).item()) <= 50:
                        p_tp = p_tp + 1
                    else:
                        p_fp = p_fp + 1
                if torch.max(yp) < thres and torch.max(xp) >= thres:
                    p_fp = p_fp + 1
                if torch.max(yp) >= thres and torch.max(xp) < thres:
                    p_fn = p_fn + 1
                if torch.max(yp) < thres and torch.max(xp) < thres:
                    p_tn = p_tn + 1
        
            progre.set_postfix({"TP": p_tp, "FP": p_fp, "TN": p_tn, "FN": p_fn})

        elif args.model_type == 'ps':
            for num in range(len(x)):
                xp = x[num,0]
                xs = x[num,1]
                yp = y[num,0]
                ys = y[num,1]
                    
                if torch.max(yp) >= thres and torch.max(xp) >= thres:
                    if abs(torch.argmax(yp).item() - torch.argmax(xp).item()) <= 50:
                        p_tp = p_tp + 1
                    else:
                        p_fp = p_fp + 1
                if torch.max(yp) < thres and torch.max(xp) >= thres:
                    p_fp = p_fp + 1
                if torch.max(yp) >= thres and torch.max(xp) < thres:
                    p_fn = p_fn + 1
                if torch.max(yp) < thres and torch.max(xp) < thres:
                    p_tn = p_tn + 1

                if torch.max(ys) >= thres and torch.max(xs) >= thres:
                    if abs(torch.argmax(ys).item() - torch.argmax(xs).item()) <= 50:
                        s_tp = s_tp + 1
                    else:
                        s_fp = s_fp + 1
                if torch.max(ys) < thres and torch.max(xs) >= thres:
                    s_fp = s_fp + 1
                if torch.max(ys) >= thres and torch.max(xs) < thres:
                    s_fn = s_fn + 1
                if torch.max(ys) < thres and torch.max(xs) < thres:
                    s_tn = s_tn + 1


    if args.model_type == 'onlyp':

        if p_tp == 0:
            p_recall = 0
            p_precision = 0
            p_f1 = 0
        else:
            p_recall = p_tp / (p_tp + p_fn)
            p_precision = p_tp / (p_tp + p_fp)
            p_f1 = 2*((p_precision * p_recall)/(p_precision+p_recall))
        
        # Write Log
        f.write('==================================================' + '\n')
        f.write('Threshold = ' + str(thres) + '\n')
        f.write('P-phase precision = ' + str(p_precision) + '\n')
        f.write('P-phase recall = ' + str(p_recall) + '\n')
        f.write('P-phase f1score = ' + str(p_f1) + '\n')
        f.write('P-phase tp = ' + str(p_tp) + '\n')
        f.write('P-phase fp = ' + str(p_fp) + '\n')
        f.write('P-phase tn = ' + str(p_tn) + '\n')
        f.write('P-phase fn = ' + str(p_fn) + '\n')
        
        print('==================================================')
        print('Threshold = ' + str(thres))
        print('P-phase precision = ' + str(p_precision))
        print('P-phase recall = ' + str(p_recall))
        print('P-phase f1score = ' + str(p_f1))
        print('P-phase tp = ' + str(p_tp))
        print('P-phase fp = ' + str(p_fp))
        print('P-phase tn = ' + str(p_tn))
        print('P-phase fn = ' + str(p_fn))

    elif args.model_type == 'ps':

        p_recall = p_tp / (p_tp + p_fn)
        p_precision = p_tp / (p_tp + p_fp)
        p_f1 = 2*((p_precision * p_recall)/(p_precision+p_recall))
        s_recall = s_tp / (s_tp + s_fn)
        s_precision = s_tp / (s_tp + s_fp)
        s_f1 = 2*((s_precision * s_recall)/(s_precision+s_recall))
        
        # Write Log
        f.write('==================================================' + '\n')
        f.write('Threshold = ' + str(thres) + '\n')
        f.write('P-phase precision = ' + str(p_precision) + '\n')
        f.write('P-phase recall = ' + str(p_recall) + '\n')
        f.write('P-phase f1score = ' + str(p_f1) + '\n')
        f.write('P-phase tp = ' + str(p_tp) + '\n')
        f.write('P-phase fp = ' + str(p_fp) + '\n')
        f.write('P-phase tn = ' + str(p_tn) + '\n')
        f.write('P-phase fn = ' + str(p_fn) + '\n')
        f.write('S-phase============================' + '\n')
        f.write('S-phase precision = ' + str(s_precision) + '\n')
        f.write('S-phase recall = ' + str(s_recall) + '\n')
        f.write('S-phase f1score = ' + str(s_f1) + '\n')
        f.write('S-phase tp = ' + str(s_tp) + '\n')
        f.write('S-phase fp = ' + str(s_fp) + '\n')
        f.write('S-phase tn = ' + str(s_tn) + '\n')
        f.write('S-phase fn = ' + str(s_fn) + '\n')
        
        print('==================================================')
        print('Threshold = ' + str(thres))
        print('P-phase precision = ' + str(p_precision))
        print('P-phase recall = ' + str(p_recall))
        print('P-phase f1score = ' + str(p_f1))
        print('P-phase tp = ' + str(p_tp))
        print('P-phase fp = ' + str(p_fp))
        print('P-phase tn = ' + str(p_tn))
        print('P-phase fn = ' + str(p_fn))
        print('S-phase============================')
        print('S-phase precision = ' + str(s_precision))
        print('S-phase recall = ' + str(s_recall))
        print('S-phase f1score = ' + str(s_f1))
        print('S-phase tp = ' + str(s_tp))
        print('S-phase fp = ' + str(s_fp))
        print('S-phase tn = ' + str(s_tn))
        print('S-phase fn = ' + str(s_fn))
    

f.close()
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Threshold time: {elapsed_time} sec")
print("=====================================================")
sys.exit()