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
import matplotlib.pyplot as plt
from model import Wav2vec_Pick
from utils import parse_arguments
import logging
logging.getLogger().setLevel(logging.WARNING)
# =========================================================================================================
# Parameter init
args = parse_arguments()
threshold = args.threshold
model_name = args.model_name
print(model_name)
ptime = 500
window = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parl = 'y'  # y,n
# =========================================================================================================
cwb_path = "/mnt/nas5/johnn9/dataset/cwbsn/"
tsm_path = "/mnt/nas5/johnn9/dataset/tsmip/"
noi_path = "/mnt/nas5/johnn9/dataset/cwbsn_noise/"
mod_path = "/mnt/nas3/johnn9/checkpoint/"
test_name = 'threshold=' + str(threshold) + '_eval'
model_path = mod_path + model_name
threshold_path = model_path + '/' + test_name + '.txt'
loadmodel = model_path + '/' + 'best_checkpoint.pt' 
image_path = model_path + '/' + test_name + '_fig'
if not os.path.isdir(image_path):
    os.mkdir(image_path)
tp_path = image_path + '/tp'
fp_path = image_path + '/fp'
fpn_path = image_path + '/fpn'
tn_path = image_path + '/tn'
fn_path = image_path + '/fn'
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
# CWBSN load
cwb = sbd.WaveformDataset(cwb_path,sampling_rate=100)
c_mask = cwb.metadata["trace_completeness"] > 3
cwb.filter(c_mask)
_, c_dev, c_test = cwb.train_dev_test()
# TSMIP load
tsm = sbd.WaveformDataset(tsm_path,sampling_rate=100)
t_mask = tsm.metadata["trace_completeness"] == 1
tsm.filter(t_mask)
_, t_dev, t_test = tsm.train_dev_test()
# NOISE load
# Combine
if args.test_set == 'test':
    if args.noise_need == 'true':        
        noise = sbd.WaveformDataset(noi_path,sampling_rate=100)
        _, n_dev, n_test = noise.train_dev_test()
        test = c_test + t_test + n_test
    elif args.noise_need == 'false':
        test = c_test + t_test
elif args.test_set == 'dev':
    if args.noise_need == 'true':        
        noise = sbd.WaveformDataset(noi_path,sampling_rate=100)
        _, n_dev, n_test = noise.train_dev_test()
        test = c_dev + t_dev + n_dev
    elif args.noise_need == 'false':
        test = c_dev + t_dev
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")
# =========================================================================================================

def label_gen(label):
    # (B,3,3000)
    label = label[:,0,:]
    label = torch.unsqueeze(label,1)
    # other = torch.ones_like(label)-label
    # label = torch.cat((label,other), dim=1)

    return label


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
            plt.subplot(515) 
            plt.plot(p_label)
            plt.title('P_label')
            plt.tight_layout()
            fignum1 = str(image)
            savepath = savepath + str(num) + '.png'
            plt.savefig(savepath)
            plt.close('all')

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
test_gene = sbg.GenericGenerator(test)
test_gene.add_augmentations(augmentations)
test_loader = DataLoader(test_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Whole model build
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
print("Testing start!!!")
start_time = time.time()
f = open(threshold_path,"w")
print("Testing: " + str(threshold) + " start!!")
p_tp,p_tn,p_fp,p_fn,p_fpn = 0,0,0,0,0
s_tp,s_tn,s_fp,s_fn = 0,0,0,0
image = 0
# Test loop
progre = tqdm(test_loader,total = len(test_loader), ncols=80)
for batch in progre:
    x = batch['X'].to(device)
    y = batch['y'].to(device)
    y = label_gen(y)
    x = model(x.to(device))

    if args.model_type == 'onlyp':
        for num in range(len(x)):
            xp = x[num,0]
            yp = y[num,0]
                
            if torch.max(yp) >= threshold and torch.max(xp) >= threshold:
                if abs(torch.argmax(yp).item() - torch.argmax(xp).item()) <= 50:
                    p_tp = p_tp + 1
                    if p_tp < 10:
                        image_save(batch,x,y,tp_path,p_tp,num)
                else:
                    p_fp = p_fp + 1
                    if p_fp < 10:
                        image_save(batch,x,y,fp_path,p_fp,num)
            if torch.max(yp) < threshold and torch.max(xp) >= threshold:
                p_fpn = p_fpn + 1
                if p_fpn < 10:
                    image_save(batch,x,y,fpn_path,p_fpn,num)
            if torch.max(yp) >= threshold and torch.max(xp) < threshold:
                p_fn = p_fn + 1
                if p_fn < 10:
                    image_save(batch,x,y,fn_path,p_fn,num)
            if torch.max(yp) < threshold and torch.max(xp) < threshold:
                p_tn = p_tn + 1
                if p_tn < 10:
                    image_save(batch,x,y,tn_path,p_tn,num)
        progre.set_postfix({"TP": p_tp, "FP": p_fp+p_fpn, "TN": p_tn, "FN": p_fn})
    
    elif args.model_type == 'ps':
        for num in range(len(x)):
            xp = x[num,0]
            xs = x[num,1]
            yp = y[num,0]
            ys = y[num,1]
                
            if torch.max(yp) >= threshold and torch.max(xp) >= threshold:
                if abs(torch.argmax(yp).item() - torch.argmax(xp).item()) <= 50:
                    p_tp = p_tp + 1
                else:
                    p_fp = p_fp + 1
            if torch.max(yp) < threshold and torch.max(xp) >= threshold:
                p_fp = p_fp + 1
            if torch.max(yp) >= threshold and torch.max(xp) < threshold:
                p_fn = p_fn + 1
            if torch.max(yp) < threshold and torch.max(xp) < threshold:
                p_tn = p_tn + 1

            if torch.max(ys) >= threshold and torch.max(xs) >= threshold:
                if abs(torch.argmax(ys).item() - torch.argmax(xs).item()) <= 50:
                    s_tp = s_tp + 1
                else:
                    s_fp = s_fp + 1
            if torch.max(ys) < threshold and torch.max(xs) >= threshold:
                s_fp = s_fp + 1
            if torch.max(ys) >= threshold and torch.max(xs) < threshold:
                s_fn = s_fn + 1
            if torch.max(ys) < threshold and torch.max(xs) < threshold:
                s_tn = s_tn + 1

        if image < 50:
            
            wave1 = batch['X'][0,0]
            wave2 = batch['X'][0,1]
            wave3 = batch['X'][0,2]
            # 创建示例波形数据、one-hot label 和 softmax 概率值
            p_predict = x[0,0]  # 长度为 9 的 one-hot label
            p_label = y[0,0]  # 长度为 9 的 softmax 概率值，这里使用随机数据代替
            s_predict = x[0,1]
            s_label = y[0,1]
            
            wave1 = wave1.detach().numpy()
            wave2 = wave2.detach().numpy()
            wave3 = wave3.detach().numpy()
            p_predict = p_predict.detach().cpu().numpy()
            p_label = p_label.detach().cpu().numpy()
            s_predict = s_predict.detach().cpu().numpy()
            s_label = s_label.detach().cpu().numpy()

            fig = plt.figure(figsize=(15,10))
            axs = fig.subplots(7,1,sharex=True,gridspec_kw={"hspace":0,"height_ratios":[1,1,1,1,1,1,1]})
            axs[0].plot(wave1)
            axs[1].plot(wave2)
            axs[2].plot(wave3)
            axs[3].plot(p_predict)
            axs[4].plot(p_label)
            axs[5].plot(s_predict)
            axs[6].plot(s_label)
            fignum1 = str(image)
            savepath = image_path + '/wav_' + fignum1 + '.png'
            plt.savefig(savepath)
            plt.close('all') 
            image = image + 1

    if args.test_mode == 'true':
        break


# 計算分數 
p_fp = p_fp+p_fpn
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
    f.write('Threshold = ' + str(threshold) + '\n')
    f.write('P-phase precision = ' + str(p_precision) + '\n')
    f.write('P-phase recall = ' + str(p_recall) + '\n')
    f.write('P-phase f1score = ' + str(p_f1) + '\n')
    f.write('P-phase tp = ' + str(p_tp) + '\n')
    f.write('P-phase fp = ' + str(p_fp) + '\n')
    f.write('P-phase tn = ' + str(p_tn) + '\n')
    f.write('P-phase fn = ' + str(p_fn) + '\n')
    # Print Log
    print('==================================================')
    print('Threshold = ' + str(threshold))
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
    f.write('Threshold = ' + str(threshold) + '\n')
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
    # Print Log
    print('==================================================')
    print('Threshold = ' + str(threshold))
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
print(f"Testing time: {elapsed_time} sec")
print("=====================================================")
sys.exit()