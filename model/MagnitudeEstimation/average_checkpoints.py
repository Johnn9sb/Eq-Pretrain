import numpy as np
import os
import re
import torch
import collections
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', type=int)
    parser.add_argument('--save_path', type=str)

    opt = parser.parse_args()

    return opt

def average_checkpoints(n, save_path):
    files = os.listdir(save_path)
    
    epoch_list = []
    for f in files:
        m = re.search(r'\d+', f)

        if m is not None:
            epoch = int(m.group())
            epoch_list.append(epoch)
    
    max_epoch = max(epoch_list)
    
    # load checkpoints
    checkpoints = []
    checkpoints.append(torch.load(os.path.join(save_path, 'model.pt'), map_location='cpu'))
    for i in range(n-1):
        checkpoints.append(torch.load(os.path.join(save_path, f"model_epoch{max_epoch-i}.pt"), map_location='cpu'))
        
    # accumulating
    params_dict = collections.OrderedDict()
    
    for idx in range(len(checkpoints)):
        model_params = checkpoints[idx]["model"]
        model_params_keys = list(model_params.keys())
        
        params_keys = model_params_keys

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
                
    # averaging
    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(n)
        else:
            averaged_params[k] //= n
         
    new_stat = checkpoints[0]
    new_stat["model"] = averaged_params
    
    return new_stat

if __name__ == '__main__':
    opt = parse_args()

    save_dir = os.path.join('./results', opt.save_path)

    new_stat = average_checkpoints(opt.n, save_dir)

    torch.save(new_stat, os.path.join(save_dir, 'averaged_checkpoint.pt'))


