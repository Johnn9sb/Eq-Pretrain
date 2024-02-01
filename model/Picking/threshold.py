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
from model import Wav2vec_Pick
from utils import parse_arguments,get_dataset
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
threshold = [0.1,0.2,0.3,0.4,0.5,0.6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================================================================
mod_path = "/mnt/nas3/johnn9/checkpoint/"
model_path = mod_path + model_name
threshold_path = model_path + '/' + test_name + '.txt'
loadmodel = model_path + '/' + 'best_checkpoint.pt' 
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
_,dev,_ = get_dataset(args)
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
    sbg.RandomWindow(windowlen=window, strategy="pad",low=250,high=5750),
    # sbg.FixedWindow(p0=3000-ptime,windowlen=3000,strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
    # sbg.DetectionLabeller(p_phases=p_dict, s_phases=s_dict),
]
dev_gene = sbg.GenericGenerator(dev)
dev_gene.add_augmentations(augmentations)
dev_loader = DataLoader(dev_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Wav2vec model load
if args.train_model == "wav2vec2":
    print(args.checkpoint_path)
    model = Wav2vec_Pick(
        device=device,
        decoder_type=args.decoder_type,
        checkpoint_path=args.checkpoint_path,
    )
elif args.train_model == "phasenet":
    model = sbm.PhaseNet(phases="PSN", norm="peak")
elif args.train_model == "eqt":
    model = sbm.EQTransformer(in_samples=window, phases='PS')

if args.parl == 'y':
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
    p_mean,p_std,p_mae = 0,0,0
    s_mean,s_std,s_mae = 0,0,0
    progre = tqdm(dev_loader, total=len(dev_loader), ncols=80)
    for batch in progre:
        p_mean_batch,p_std_batch,p_mae_batch = 0,0,0
        s_mean_batch,s_std_batch,s_mae_batch = 0,0,0
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        # y = label_gen(y.to(device))
        x = model(x.to(device))
        if args.task == 'pick':
            if train_model == 'eqt':
                x_tensor = torch.empty(2,len(y),window)
                for index, item in enumerate(x):
                    x_tensor[index] = item
                    if index == 1:
                        break
                x = x_tensor.permute(1,0,2)
                x = x.to(device)
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
                
                p_mean_now = torch.mean(xp - yp)
                p_mean_batch = p_mean_batch + p_mean_now.item()
                p_std_now = torch.std(xp - yp)
                p_std_batch = p_std_batch + p_std_now.item()
                p_mae_now = torch.mean(torch.abs(xp - yp))
                p_mae_batch = p_mae_batch + p_mae_now
                s_mean_now = torch.mean(xs - ys)
                s_mean_batch = s_mean_batch + s_mean_now.item()
                s_std_now = torch.std(xs - ys)
                s_std_batch = s_std_batch + s_std_now.item()
                s_mae_now = torch.mean(torch.abs(xs - ys))
                s_mae_batch = s_mae_batch + s_mae_now
            
            p_mean = p_mean + (p_mean_batch / args.batch_size)
            p_std = p_std + (p_std_batch / args.batch_size)
            p_mae = p_mae + (p_mae_batch / args.batch_size)
            s_mean = s_mean + (s_mean_batch / args.batch_size)
            s_std = s_std + (s_std_batch / args.batch_size)
            s_mae = s_mae + (s_mae_batch / args.batch_size)

            progre.set_postfix({"pTP": p_tp, "pFP": p_fp, "pTN": p_tn, "pFN": p_fn})

    p_mean = p_mean / len(dev_loader)
    p_std = p_std / len(dev_loader)
    p_mae = p_mae / len(dev_loader)
    s_mean = s_mean / len(dev_loader)
    s_std = s_std / len(dev_loader)
    s_mae = s_mae / len(dev_loader)

    if args.task == 'pick':

        if p_tp == 0:
            p_recall = 0
            p_precision = 0
            p_f1 = 0
        else:
            p_recall = p_tp / (p_tp + p_fn)
            p_precision = p_tp / (p_tp + p_fp)
            p_f1 = 2*((p_precision * p_recall)/(p_precision+p_recall))
        if s_tp == 0:
            s_recall = 0
            s_precision = 0
            s_f1 = 0
        else:
            s_recall = s_tp / (s_tp + s_fn)
            s_precision = s_tp / (s_tp + s_fp)
            s_f1 = 2*((s_precision * s_recall)/(s_precision+s_recall))
        # Write Log
        f.write('==================================================' + '\n')
        f.write('Threshold = ' + str(thres) + '\n')
        f.write('P-phase precision = ' + str(p_precision) + '\n')
        f.write('P-phase recall = ' + str(p_recall) + '\n')
        f.write('P-phase f1score = ' + str(p_f1) + '\n')
        f.write('P-phase mean = ' + str(p_mean) + '\n')
        f.write('P-phase std = ' + str(p_std) + '\n')
        f.write('P-phase mae = ' + str(p_mae) + '\n')
        f.write('P-phase tp = ' + str(p_tp) + '\n')
        f.write('P-phase fp = ' + str(p_fp) + '\n')
        f.write('P-phase tn = ' + str(p_tn) + '\n')
        f.write('P-phase fn = ' + str(p_fn) + '\n')
        f.write('==================================================' + '\n')
        f.write('S-phase precision = ' + str(s_precision) + '\n')
        f.write('S-phase recall = ' + str(s_recall) + '\n')
        f.write('S-phase f1score = ' + str(s_f1) + '\n')
        f.write('S-phase mean = ' + str(s_mean) + '\n')
        f.write('S-phase std = ' + str(s_std) + '\n')
        f.write('S-phase mae = ' + str(s_mae) + '\n')
        f.write('S-phase tp = ' + str(s_tp) + '\n')
        f.write('S-phase fp = ' + str(s_fp) + '\n')
        f.write('S-phase tn = ' + str(s_tn) + '\n')
        f.write('S-phase fn = ' + str(s_fn) + '\n')
        
        print('==================================================')
        print('Threshold = ' + str(thres))
        print('P-phase precision = ' + str(p_precision))
        print('P-phase recall = ' + str(p_recall))
        print('P-phase f1score = ' + str(p_f1))
        print('P-phase mean = ' + str(p_mean))
        print('P-phase std = ' + str(p_std))
        print('P-phase mae = ' + str(p_mae))
        print('P-phase tp = ' + str(p_tp))
        print('P-phase fp = ' + str(p_fp))
        print('P-phase tn = ' + str(p_tn))
        print('P-phase fn = ' + str(p_fn))
        print('==================================================')
        print('Threshold = ' + str(thres))
        print('S-phase precision = ' + str(s_precision))
        print('S-phase recall = ' + str(s_recall))
        print('S-phase f1score = ' + str(s_f1))
        print('S-phase mean = ' + str(s_mean))
        print('S-phase std = ' + str(s_std))
        print('S-phase mae = ' + str(s_mae))
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