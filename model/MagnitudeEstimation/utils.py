import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

import numpy as np
import sys
import torch
import torch.nn.functional as F
import argparse
import re
import pandas as pd
from argparse import Namespace
from model import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    # basic hyperparameters
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='none')
    parser.add_argument('--w2v_path', type=str, default='../../checkpoint.pt')
    parser.add_argument('--load_specific_model', type=str, default='model')
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--patience', type=float, default=7)
    parser.add_argument('--noam', type=bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=1500)
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--config_path', type=str, default='none')
    parser.add_argument('--p_timestep', type=int, default=500)

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--instrument', type=str, default='all')
    parser.add_argument('--location', type=int, default=-1)
    parser.add_argument("--filter_instance", type=bool, default=False)
    parser.add_argument('--without_noise', type=bool, default=False)
    parser.add_argument('--wavelength', type=int, default=3000)
    parser.add_argument('--epidis', type=float, default=-1)
    parser.add_argument('--snr', type=float, default=-1)

    # data augmentations
    parser.add_argument('--gaussian_noise_prob', type=float, default=0.5)
    parser.add_argument('--channel_dropout_prob', type=float, default=0.3)
    parser.add_argument('--adding_gap_prob', type=float, default=0.2)

    # seisbench options
    parser.add_argument('--model_opt', type=str, default='w2v')
    parser.add_argument('--loss_weight', type=float, default=10)
    parser.add_argument('--dataset_opt', type=str, default='cwb')
    parser.add_argument('--loading_method', type=str, default='full')
    
    # custom hyperparameters
    parser.add_argument('--decoder_type', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.1)

    opt = parser.parse_args()

    # load the config 
    if opt.config_path != 'none':
        f = open(opt.config_path, 'r')
        config = json.load(f)

        opt = vars(opt)
        opt.update(config)

        opt = Namespace(**opt)

    return opt

def load_dataset(opt):
    cwbsn, tsmip, stead, cwbsn_noise, instance = 0, 0, 0, 0, 0
    
    if opt.dataset_opt == 'instance' or opt.dataset_opt == 'all':
        print('loading INSTANCE')
        kwargs={'download_kwargs': {'basepath': '/home/weiwei/disk4/seismic_datasets/'}}
        instance = sbd.InstanceCountsCombined(**kwargs)

        instance = apply_filter(instance, isINSTANCE=True, filter_instance=opt.filter_instance,
                            epidis=opt.epidis, snr=opt.snr, without_noise=opt.without_noise)

    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)

        stead = apply_filter(stead, isStead=True, epidis=opt.epidis, snr=opt.snr, without_noise=opt.without_noise)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method=opt.loading_method, **kwargs)
        cwbsn = apply_filter(cwbsn, isCWBSN=True, level=opt.level, instrument=opt.instrument, location=opt.location,
                            epidis=opt.epidis, snr=opt.snr, without_noise=opt.without_noise)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all':
        # TSMIP
        print('loading TSMIP') 
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method=opt.loading_method, sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        tsmip = apply_filter(tsmip, instrument=opt.instrument, location=opt.location,
                            epidis=opt.epidis, snr=opt.snr, without_noise=opt.without_noise)

    if not opt.without_noise and (opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all'):
        # CWBSN noise
        print('loading CWBSN noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/disk4/weiwei/seismic_datasets/CWB_noise/'}}
        cwbsn_noise = sbd.CWBSN_noise(**kwargs)
        
        cwbsn_noise = apply_filter(cwbsn_noise, instrument=opt.instrument, isNoise=True, location=opt.location,
                            epidis=opt.epidis, snr=opt.snr, without_noise=opt.without_noise)

        print('traces: ', len(cwbsn_noise))

    return cwbsn, tsmip, stead, cwbsn_noise, instance

def apply_filter(data, isCWBSN=False, level=-1, isStead=False, isNoise=False, instrument='all', location=-1, isINSTANCE=False, filter_instance=False
                , epidis=-1, snr=-1, without_noise=False):
    # Apply filter on seisbench.data class

    print('original traces: ', len(data))
    
    # 只選波型完整的 trace
    if not isStead and not isNoise and not isINSTANCE:
        if isCWBSN:
            if level != -1:
                complete_mask = data.metadata['trace_completeness'] == level
            else:
                complete_mask = np.logical_or(data.metadata['trace_completeness'] == 3, data.metadata['trace_completeness'] == 4)
        else:
            complete_mask = data.metadata['trace_completeness'] == 1

        # 只選包含一個事件的 trace
        single_mask = data.metadata['trace_number_of_event'] == 1

        # making final mask
        mask = np.logical_and(single_mask, complete_mask)
        data.filter(mask)

    # 篩選儀器 location
    if location != -1:
        location_mask = np.logical_or(data.metadata['station_location_code'] == location, data.metadata['station_location_code'] == str(location))
        data.filter(location_mask)

    # 篩選儀器 channel
    if instrument != 'all':
        instrument_mask = data.metadata["trace_channel"] == instrument
        data.filter(instrument_mask)

    # 只取高品質的波型 (for INSTANCE)
    if isINSTANCE and filter_instance:
        p_weight_mask = data.metadata['path_weight_phase_location_P'] >= 50
        eqt_mask = np.logical_and(data.metadata['trace_EQT_number_detections'] == 1, data.metadata['trace_EQT_P_number'] == 1)
        instance_mask = np.logical_and(p_weight_mask, eqt_mask)
        data.filter(instance_mask)

    # 篩選 STEAD noise
    if isStead and without_noise:
        seis_mask = data.metadata['trace_category'] == 'earthquake_local'
        data.filter(seis_mask)

    # 篩選 trace 的 epicentral distance
    if epidis != -1:
        try:
            if isStead:
                epidis_mask = data.metadata['source_distance_km'] <= epidis
                data.filter(epidis_mask)
            else:
                epidis_mask = data.metadata['path_ep_distance_km'] <= epidis
                data.filter(epidis_mask)
            
        except Exception as e:
            print('Skipping filter the epicentral distance of trace !')
            print(e)

    # 篩選 trace 的 SNRs
    if snr != -1:
        try:
            if isStead:
                snr_mask = create_stead_snr_mask(data.metadata, snr)
                data.metadata.reset_index(drop=True, inplace=True)
                data.filter(snr_mask)
            else:
                snr_mask = data.metadata['trace_Z_snr_db'] >= snr
                data.filter(snr_mask)
        except Exception as e:
            print('Skipping filter the SNR of trace !')
            print(e)

    print('filtered traces: ', len(data))

    return data

def basic_augmentations(opt, ptime=None, test=False):
    # basic augmentations:
    #   1) Windowed around p-phase pick
    #   2) Random cut window, wavelen=3000
    #   3) Filter 
    #   4) Normalize: demean, zscore,
    #   5) Change dtype to float32
    #   6) Probabilistic: gaussian function
    
    if opt.dataset_opt == 'instance':
        p_phases = 'trace_P_arrival_sample'
    else:
        p_phases = 'trace_p_arrival_sample'
    
    phase_dict = [p_phases]

    if test:
        if opt.dataset_opt == 'stead':
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                sbg.ChangeDtype(np.float32),
                sbg.Magnitude(),
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.Magnitude(),
            ]
    else:
        if opt.dataset_opt == 'stead':
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=1000, high=5901),
                sbg.ChangeDtype(np.float32),
                sbg.Magnitude(),
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=1000, high=5901),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.ChangeDtype(np.float32),
                sbg.Magnitude(),
            ]
    
    return augmentations

def load_model(opt, device):
    assert opt.model_opt != None, "Choose one of the model in seisbench."

    if opt.model_opt == 'w2v':
        model = Wav2Vec_Mag(decoder_type=opt.decoder_type, device=device, wavelength=opt.wavelength, checkpoint_path=opt.w2v_path)
    elif opt.model_opt == 'magnet':
        model = MagNet()

    return model.to(device)

def loss_fn(opt, pred, gt, device):
    criterion = nn.MSELoss()
    loss = criterion(pred, gt.type(torch.FloatTensor).to(device))

    return loss

def create_stead_snr_mask(df, snr):
    def add_comma(match):
        return match.group(0) + ','

    stead_snr = df['trace_snr_db'].tolist()

    mask = []
    for s in stead_snr:
        try:
            s = re.sub(r'\[[0-9\.\s]+\]', add_comma, s)
            s = re.sub(r'([0-9\.]+)', add_comma, s)
            Zsnr = eval(s)[0][0]

            if Zsnr >= snr:
                mask.append(True)
            else:
                mask.append(False)
        except Exception as e:
            try:
                Zsnr = eval(s)[0]

                if Zsnr >= snr:
                    mask.append(True)
                else:
                    mask.append(False)
            except:
                mask.append(True)

    # 暴力補足剩下的 mask
    print("mask: ", len(mask))
    print('df: ', len(df))
    diff = len(df) - len(mask)
    for i in range(diff):
        mask.append(True)

    return pd.Series(mask)
