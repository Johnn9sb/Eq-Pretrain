import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append('../')
from ds_wav2vec2 import Wav2Vec2Model,Wav2Vec2Config


class Wav2vec_Pick(nn.Module):
    def __init__(self, args, decoder_type, device, checkpoint_path='../../checkpoint.pt'):
        super().__init__()
        self.w2v = Wav2Vec2Model(Wav2Vec2Config)
        self.args = args
        if checkpoint_path != 'None':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.w2v.load_state_dict(checkpoint['model'], strict=True)
        self.decoder_type = decoder_type
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if decoder_type == 'linear':
            self.Li_1 = nn.Sequential(
                nn.Linear(in_features=750, out_features=3000),
                nn.BatchNorm1d(num_features=768),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_2 = nn.Sequential(
                nn.Linear(in_features=768, out_features=2),
            )
        
        elif decoder_type == 'cnn':
            self.cnn_1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=768,
                    out_channels=256,
                    kernel_size=7,
                    padding='same',
                ),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=5,
                    padding='same',
                ),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_3 = nn.Sequential(
                nn.Conv1d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=5,
                    padding='same',
                ),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_4 = nn.Sequential(
                nn.Conv1d(
                    in_channels=256,
                    out_channels=2,
                    kernel_size=11,
                    padding='same',
                ),
                nn.Sigmoid(),
            )
        
        elif decoder_type == 'low_linear':
            self.Li_1 = nn.Sequential(
                nn.Linear(in_features=192, out_features=64, bias=False),
                nn.BatchNorm1d(num_features=750),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_2 = nn.Sequential(
                nn.Linear(in_features=750, out_features=3000),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.Li_3 = nn.Sequential(
                nn.Linear(in_features=64, out_features=2),
            )


    def forward(self, x):
        # Wav2Vec frozen
        if self.args.freeze == 'n':
            x = self.w2v(x)
        else:
            with torch.no_grad():
                x = self.w2v(x)
        if self.decoder_type == 'linear':
            x = self.Li_1(x.permute(0,2,1))
            x = self.Li_2(x.permute(0,2,1))
            x = self.sigmoid(x.permute(0,2,1))
        
        elif self.decoder_type == 'cnn':
            x = self.cnn_1(x.permute(0,2,1))
            x = self.upsample(x)
            x = self.cnn_2(x)
            x = self.upsample(x)
            x = self.cnn_3(x)
            x = self.cnn_4(x)

        elif self.decoder_type == 'low_linear':
            x = x.view(x.size(0), x.size(1), -1, 4)
            x = x.sum(dim=-1)
            x = self.Li_1(x)
            x = self.Li_2(x.permute(0,2,1))
            x = self.Li_3(x.permute(0,2,1))
            x = self.sigmoid(x.permute(0,2,1))

        return x
