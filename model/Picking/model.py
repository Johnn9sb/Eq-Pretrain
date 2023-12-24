import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append('../')
from wav2vec2 import Wav2Vec2Model,Wav2Vec2Config


class Wav2vec_Pick(nn.Module):
    def __init__(self, decoder_type, device, checkpoint_path='../../checkpoint.pt'):
        super().__init__()
        self.w2v = Wav2Vec2Model(Wav2Vec2Config)
        if checkpoint_path != 'None':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.w2v.load_state_dict(checkpoint['model'], strict=True)
        self.decoder_type = decoder_type
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        if decoder_type == 'linear':
            self.Li_1 = nn.Sequential(
                nn.Linear(in_features=750, out_features=1500),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_1_5 = nn.Sequential(
                nn.Linear(in_features=128, out_features=128),
                nn.BatchNorm1d(num_features=1500),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_2 = nn.Sequential(
                nn.Linear(in_features=1500, out_features=3000),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_3 = nn.Sequential(
                nn.Linear(in_features=128, out_features=1),
                nn.BatchNorm1d(num_features=3000),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_out = nn.Sequential(
                nn.Linear(in_features=3000, out_features=3000),
                nn.Sigmoid(),
            )
        
        # elif decoder_type == 'cnn':


    def forward(self, x):
        # Wav2Vec frozen
        with torch.no_grad():
            x = self.w2v(x)
        # print(x.shape)
        if self.decoder_type == 'linear':
            x = self.Li_1(x.permute(0,2,1))
            x = self.Li_1_5(x.permute(0,2,1))
            x = self.Li_2(x.permute(0,2,1))
            x = self.Li_3(x.permute(0,2,1))
            x = self.Li_out(x.permute(0,2,1))

        return x
