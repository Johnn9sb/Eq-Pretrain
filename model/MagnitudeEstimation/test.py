import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import pickle
import json
import time
import bisect
import requests
from tqdm import tqdm
from argparse import Namespace
from utils import *
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

def toLine(save_path, precision, recall, fscore, mean, variance):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    message = save_path + ' -> precision: ' + str(precision) + ', recall: ' + str(recall) + ', fscore: ' + str(fscore) + ', mean: ' + str(mean) + ', variance: ' + str(variance)
    
    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
        }
        payload = {
            'message': message
        }
        response = requests.request(
            "POST",
            url,
            headers=headers,
            data=payload
        )
        if response.status_code == 200:
            print(f"Success -> {response.text}")
    except Exception as e:
        print(e)

def set_generators(opt, ptime=None):
    cwbsn, tsmip, stead, cwbsn_noise, instance = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        stead_dev, stead_test = stead.dev(), stead.test()
        cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()
        instance_train, instance_dev, _ = instance.train_dev_test()

        train = cwbsn_train + tsmip_train + stead_train + cwbsn_noise_train + instance_train
        dev = cwbsn_dev + tsmip_dev + stead_dev + cwbsn_noise_dev + instance_dev
    elif opt.dataset_opt == 'cwbsn':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        stead_dev, stead_test = stead.dev(), stead.test()

        dev = cwbsn_dev + stead_dev
        test = cwbsn_test + stead_test
    elif opt.dataset_opt == 'tsmip':
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()

        dev = tsmip_dev
        test = tsmip_test
    elif opt.dataset_opt == 'stead':
        _, dev, test = stead.train_dev_test()
    elif opt.dataset_opt == 'instance':
        _, dev, test = instance.train_dev_test()
    elif opt.dataset_opt == 'cwb':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()

        if not opt.without_noise:
            cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()

            dev = cwbsn_dev + tsmip_dev + cwbsn_noise_dev
            test = cwbsn_test + tsmip_test + cwbsn_noise_test
        else:
            dev = cwbsn_dev + tsmip_dev
            test = cwbsn_test + tsmip_test

    print(f'total traces -> dev: {len(dev)}, test: {len(test)}')

    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    # set generator with or without augmentations
    augmentations = basic_augmentations(opt, test=True, ptime=opt.p_timestep)

    dev_generator.add_augmentations(augmentations)
    test_generator.add_augmentations(augmentations)

    return dev_generator, test_generator

def inference(opt, model, test_loader, device):
    # 先把整個 test set 的預測結果都跑完一遍

    pred = []
    gt = []

    model.eval()
    with tqdm(test_loader) as epoch:
        idx = 0
        for data in epoch:  
            idx += 1        

            with torch.no_grad():
                out = model(data['X'].to(device)).cpu()
                target = data['mag']

                pred += out
                gt += target

    return pred, gt

def score(pred, gt, output_dir, f, status):
    total_diff = np.abs(np.array(pred)-np.array(gt))
    total_mae = total_diff.mean()
    total_mse = np.power(total_diff, 2).mean()
    total_std = np.std(total_diff)
    total_r2 = r2_score(pred, gt)

    noise_mask = np.array(gt) == 0
    without_noise_diff = np.abs(np.array(pred)[~noise_mask]-np.array(gt)[~noise_mask])
    without_noise_mae = without_noise_diff.mean()
    without_noise_mse = np.power(without_noise_diff, 2).mean()
    without_noise_std = np.std(without_noise_diff)
    without_noise_r2 = r2_score(np.array(pred)[~noise_mask], np.array(gt)[~noise_mask])

    mag4_mask = np.array(gt) >= 4.0
    mag4_diff = np.abs(np.array(pred)[mag4_mask]-np.array(gt)[mag4_mask])
    mag4_mae = mag4_diff.mean()
    mag4_mse = np.power(mag4_diff, 2).mean()
    mag4_std = np.std(mag4_diff)
    mag4_noise_r2 = r2_score(np.array(pred)[mag4_mask], np.array(gt)[mag4_mask])

    logging.info(f"Total -> MAE: {round(total_mae, 4)}, MSE: {round(total_mse, 4)}, std: {round(total_std, 4)}, R2score: {round(total_r2, 4)}, Count: {total_diff.shape}\n")
    logging.info(f"w/o Noise -> MAE: {round(without_noise_mae, 4)}, MSE: {round(without_noise_mse, 4)}, std: {round(without_noise_std, 4)}, R2score: {round(without_noise_r2, 4)}, Count: {without_noise_diff.shape}\n")
    logging.info(f"MAG4 -> MAE: {round(mag4_mae, 4)}, MSE: {round(mag4_mse, 4)}, std: {round(mag4_std, 4)}, R2score: {round(mag4_noise_r2, 4)}, Count: {mag4_diff.shape}\n")

    f.write(f"Total -> MAE: {round(total_mae, 4)}, MSE: {round(total_mse, 4)}, std: {round(total_std, 4)}, R2score: {round(total_r2, 4)}, Count: {total_diff.shape}\n")
    f.write(f"w/o Noise -> MAE: {round(without_noise_mae, 4)}, MSE: {round(without_noise_mse, 4)}, std: {round(without_noise_std, 4)}, R2score: {round(without_noise_r2, 4)}, Count: {without_noise_diff.shape}\n")
    f.write(f"MAG4 -> MAE: {round(mag4_mae, 4)}, MSE: {round(mag4_mse, 4)}, std: {round(mag4_std, 4)}, R2score: {round(mag4_noise_r2, 4)}, Count: {mag4_diff.shape}\n")

    with open(os.path.join(output_dir, f'{status}_pred.pkl'), 'wb') as f:
        pickle.dump(pred, f)
    with open(os.path.join(output_dir, f'{status}_gt.pkl'), 'wb') as f:
        pickle.dump(gt, f)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    model_dir = output_dir
    output_dir = os.path.join(output_dir, 'score')

    subpath = 'result_' + str(opt.p_timestep)
    if opt.level != -1:
        subpath = subpath + '_' + str(opt.level)
    if opt.load_specific_model != 'None':
        subpath = subpath + '_' + opt.load_specific_model
    if opt.instrument != 'all':
        subpath = subpath + '_' + opt.instrument
        output_dir = f"{output_dir}_{opt.instrument}"
    if opt.location != -1:
        subpath = subpath + '_' + str(opt.location)
        output_dir = f"{output_dir}_{opt.location}"
    if opt.epidis != -1:
        subpath = subpath + '_' + str(opt.epidis)
        output_dir = f"{output_dir}_{opt.epidis}"
    if opt.snr != -1:
        subpath = subpath + '_' + str(opt.snr)
        output_dir = f"{output_dir}_{opt.snr}"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subpath = subpath + '.log'
    print('logpath: ', subpath)
    log_path = os.path.join(output_dir, subpath)

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    print(opt.save_path)

    with open(log_path, 'a') as f:
        # 設定 device (opt.device = 'cpu' or 'cuda:X')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load datasets
        print('loading datasets')
        dev_generator, test_generator = set_generators(opt)
        dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        
        logging.info('dev: %d, test: %d' %(len(dev_generator), len(test_generator)))
        f.write('dev: %d, test: %d\n' %(len(dev_generator), len(test_generator)))

        # load model
        model = load_model(opt, device)

        if opt.load_specific_model != 'None':
            print('loading ', opt.load_specific_model)
            model_path = os.path.join(model_dir, opt.load_specific_model+'.pt')
        else:
            print('loading last checkpoint')
            model_path = os.path.join(model_dir, 'checkpoint_last.pt')

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        # find the best criteria
        print('Inference on validation set')
        logging.info('======================================================')
        f.write('======================================================\n')
        logging.info('Inference on validation set')
        f.write('Inference on validation set\n')
        pred, gt = inference(opt, model, dev_loader, device)
        score(pred, gt, output_dir, f, 'dev')

        print('Inference on testing set')
        logging.info('======================================================')
        f.write('======================================================\n')
        logging.info('Inference on testing set')
        f.write('Inference on testing set\n')
        pred, gt = inference(opt, model, test_loader, device)
        score(pred, gt, output_dir, f, 'test')
        logging.info('======================================================')
        f.write('======================================================\n')

        f.close()
