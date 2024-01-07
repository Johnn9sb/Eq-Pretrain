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
logging.getLogger().setLevel(logging.WARNING)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# =========================================================================================================
# main
args = parse_arguments()
# ptime = 500
model_name = args.model_name
method = '12d128'  # 1st, 2nd, 3rd, cnn3, 12d64, 12d128, 6d128
decoder_lr = 0.0005
window = 3000
parl = 'y'  # y,n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name)
# =========================================================================================================
cwb_path = "/mnt/disk2/johnn9/dataset/cwbsn/"
tsm_path = "/mnt/disk2/johnn9/dataset/tsmip/"
noi_path = "/mnt/disk2/johnn9/dataset/cwbsn_noise/"
mod_path = "/mnt/disk1/johnn9/checkpoint/"
model_path = mod_path + model_name
checkpoint = model_path+'/best_checkpoint.pt'
if not os.path.isdir(model_path):
    os.mkdir(model_path)
score_path = model_path + '/' + model_name + '.txt'
print("Init Complete!!!")
# =========================================================================================================
# DataLoad
start_time = time.time()
cwb = sbd.WaveformDataset(cwb_path,sampling_rate=100)
c_mask = cwb.metadata["trace_completeness"] == 4
cwb.filter(c_mask)
c_train, c_dev, _ = cwb.train_dev_test()
tsm = sbd.WaveformDataset(tsm_path,sampling_rate=100)
t_mask = tsm.metadata["trace_completeness"] == 1
tsm.filter(t_mask)
t_train, t_dev, _ = tsm.train_dev_test()
if args.noise_need == 'true':
    noise = sbd.WaveformDataset(noi_path,sampling_rate=100)
    n_train, n_dev, _ = noise.train_dev_test()
    train = c_train + t_train + n_train
    dev = c_dev + t_dev + n_dev
elif args.noise_need == 'false':
    train = c_train + t_train
    dev = c_dev + t_dev
train_len = len(train)
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")
# =========================================================================================================
# Funtion
start_time = time.time()
def loss_fn(x,y,args):
    
    if args.train_model == 'wav2vec2':
        weight = torch.ones_like(y)
        weight[y > 0] = args.weight
        x = x.to(torch.float16)
        y = y.to(torch.float16)
        loss_cal = nn.BCELoss(weight=weight)
        loss = loss_cal(x, y)
    elif args.train_model == 'eqt':
        x = torch.unsqueeze(x[0], 1)
        print(x.shape)
        print(y.shape)
        x = x.to(torch.float16)
        y = y.to(torch.float16)
        loss = F.binary_cross_entropy_with_logits(x, y)
    elif args.train_model == 'phasenet':
        loss = y * torch.log(x + 1e-5)
        loss = loss.mean(-1).sum(-1)
        loss = loss.mean()
        loss = -loss
    return loss

def label_gen(label,args):
    # (B,3,3000)
    if args.train_model == 'wav2vec2' or args.train_model == 'eqt':
        label = label[:,0,:]
        label = torch.unsqueeze(label,1)
    # other = torch.ones_like(label)-label
    # label = torch.cat((label,other), dim=1)
    return label

def train_loop(dataloader,win_len,args):
    num_batches = len(dataloader)
    train_loss = 0
    train_loss = float(train_loss)
    progre = tqdm(enumerate(dataloader),total=len(dataloader),ncols=80)
    for batch_id, batch in progre:
        # General 
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        y = label_gen(y.to(device),args)
        batch_size = len(x)
        # Forward
        x = model(x.to(device))
        loss = loss_fn(x,y,args)
        progre.set_postfix({'Loss': f'{loss.item():.5f}'})
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if args.test_mode == 'true':
            break
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_loss = train_loss / num_batches
    print(f"Train avg loss: {train_loss:>8f} \n")

def test_loop(dataloader,win_len,args):

    num_batches = len(dataloader)
    test_loss = 0
    test_loss = float(test_loss)
    progre = tqdm(dataloader,total=len(dataloader), ncols=80)
    for batch in progre:
        x = batch['X'].to(device)
        y = batch['y'].to(device)
        y = label_gen(y.to(device),args)
        with torch.no_grad():
            x = model(x.to(device))
        test_loss1 = loss_fn(x,y,args).item()
        test_loss = test_loss + test_loss1
        progre.set_postfix({'Test': f'{test_loss:.4f}'})
        
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if args.test_mode == 'true':
                break
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_loss = test_loss / num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")
    return test_loss

def lr_lambda(epoch):
    return 0.95 ** epoch

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
train_gene = sbg.GenericGenerator(train)
train_gene.add_augmentations(augmentations)
train_loader = DataLoader(train_gene,batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
dev_gene = sbg.GenericGenerator(dev)
dev_gene.add_augmentations(augmentations)
dev_loader = DataLoader(dev_gene,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)
print("Dataloader Complete!!!")
# =========================================================================================================
# Wav2vec model load
if args.train_model == "wav2vec2":
    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
        print(checkpoint_path)
        model = Wav2vec_Pick(
            device=device,
            decoder_type=args.decoder_type,
            checkpoint_path=checkpoint_path,
        )
    else:
        model = Wav2vec_Pick(
            device=device,
            decoder_type=args.decoder_type,
        )
elif args.train_model == "phasenet":
    model = sbm.PhaseNet(phases="PSN", norm="peak")
elif args.train_model == "eqt":
    model = sbm.EQTransformer(in_samples=window, phases='PS')

if parl == 'y':
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_indices = list(range(num_gpus))
    model = DataParallel(model, device_ids=gpu_indices)
if args.resume == 'true':
    model.load_state_dict(torch.load(checkpoint))
model.to(device)
model.cuda(); 
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=decoder_lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
# Training
testloss_log = []
lowest_loss = float('inf')
save_point = 0
print("Training start!!!")
start_time = time.time()
i = 0
for t in range(args.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    Epoch_time = time.time()
    testloss_log.append(t)
    # Train loop for one epoch
    train_loop(train_loader,window,args)
    scheduler.step()
    now_loss = test_loop(dev_loader,window,args)
    testloss_log.append(now_loss)
    
    torch.save(model.state_dict(),model_path+'/last_checkpoint.pt')
    if(now_loss < lowest_loss):
        lowest_loss = now_loss
        save_point = 0 
        lowest_epoch = t
        torch.save(model.state_dict(),model_path+'/best_checkpoint.pt')
    else:
        save_point = save_point + 1
    if(save_point > args.early_stop):
        print("Early Stop!!!")
        break
    
    Epoend_time = time.time()
    elapsed_time = Epoend_time - Epoch_time
    print("=====================================================")
    print(f"Epoch time: {elapsed_time} sec")
    print("=====================================================")
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if args.test_mode == 'true':
        break
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Training time: {elapsed_time} sec")
print("=====================================================")
# =========================================================================================================
f = open(score_path,'w')
testloss = list(map(str,testloss_log))
f.write('The most low epoch:'+str(lowest_epoch)+'\n')
for line in testloss:
    f.write(line+'\n')
f.close()
sys.exit()