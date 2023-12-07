import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append('../')
from wav2vec2 import Wav2Vec2Model,Wav2Vec2Config

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000, return_vec=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.return_vec = return_vec

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.return_vec: 
            # x: (batch_size*num_windows, window_size, input_dim)
            x = x[:] + self.pe.squeeze()

            return self.dropout(x)
        else:
            return self.pe.squeeze()

class Wav2Vec_Mag(nn.Module):
    def __init__(self, decoder_type, wavelength, device, checkpoint_path='../../checkpoint.pt'):
        super(Wav2Vec_Mag, self).__init__()

        print("Loading pretrained Wav2Vec...")
        self.w2v = Wav2Vec2Model(Wav2Vec2Config)

        if checkpoint_path != 'None':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.w2v.load_state_dict(checkpoint['model'], strict=True)

        for param in self.w2v.parameters():
            param.requires_grad = False
        print("Finish loading pretrained Wav2Vec...")
        
        self.dropout = nn.Dropout(0.1)
        self.decoder_type = decoder_type
        if decoder_type == 'linear':
            self.decoder_time = nn.Linear(wavelength//2, 1)
            self.decoder_dim = nn.Linear(128, 1)
        elif decoder_type == 'CNN_Linear':
            self.conv = nn.Sequential(nn.Conv1d(128, 96, kernel_size=7, padding='same'),
                                   nn.BatchNorm1d(96),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),
                                   nn.Conv1d(96, 64, kernel_size=5, padding='same'),
                                   nn.BatchNorm1d(64),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),
                                   nn.Conv1d(64, 32, kernel_size=3, padding='same'),
                                   nn.BatchNorm1d(32),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),)
            
            self.flatten = nn.Flatten()
            self.out = nn.Linear(736, 1)
        elif decoder_type == 'CNN_LSTM':
            self.conv = nn.Sequential(nn.Conv1d(128, 64, kernel_size=3, padding='same'),
                                    nn.Dropout(0.1),
                                    nn.MaxPool1d(4),
                                    nn.Conv1d(64, 32, kernel_size=3, padding='same'),
                                    nn.Dropout(0.1),
                                    nn.MaxPool1d(4),)
            
            self.lstm = nn.LSTM(input_size=32, hidden_size=100, bidirectional=True, batch_first=True, num_layers=1)
            
            self.linear = nn.Linear(200, 1) 

    def forward(self, wave):
        # wave: (batch, 1500, 128)

        with torch.no_grad():
            rep = self.w2v(wave)

        if self.decoder_type == 'linear':
            rep_time_reduction = self.decoder_time(rep.permute(0,2,1)).permute(0,2,1)
            out = self.decoder_dim(rep_time_reduction)

        elif self.decoder_type == 'CNN_Linear':
            cnn_out = self.conv(rep.permute(0,2,1))
            out = self.flatten(cnn_out)
            out = self.out(out)
        elif self.decoder_type == 'CNN_LSTM':
            cnn_out = self.conv(rep.permute(0,2,1)).permute(0,2,1)
            lstm_out, _ = self.lstm(cnn_out)
            out = self.linear(lstm_out)[:, -1, :]

        return out.squeeze()

class MagNet(nn.Module):
    def __init__(self,):
        super(MagNet, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv1d(3, 64, kernel_size=3, padding='same'),
                                  nn.Dropout(0.2),
                                  nn.MaxPool1d(4),
                                  nn.Conv1d(64, 32, kernel_size=3, padding='same'),
                                  nn.Dropout(0.2),
                                  nn.MaxPool1d(4),)
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=100, bidirectional=True, batch_first=True, num_layers=1)
        
        self.linear = nn.Linear(200, 1)
        
    def forward(self, wf):
        out = self.conv(wf)
        
        out, _ = self.lstm(out.permute(0,2,1))
        
        last_timestep = out[:, -1, :]
        out = self.linear(last_timestep)
        
        return out.squeeze()
