import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.append('../')
# from wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
# from ws_wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
# from data2vec1 import Data2VecAudioModel, Data2VecAudioConfig
from ws_data2vec import Data2VecAudioModel, Data2VecAudioConfig

class Wav2Vec_Mag(nn.Module):
    def __init__(self, decoder_type, wavelength, device, checkpoint_path='../../checkpoint.pt'):
        super(Wav2Vec_Mag, self).__init__()

        print("Loading pretrained Wav2Vec...")
        # self.w2v = Wav2Vec2Model(Wav2Vec2Config)
        self.w2v = Data2VecAudioModel(Data2VecAudioConfig)

        if checkpoint_path != 'None':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.w2v.load_state_dict(checkpoint['model'], strict=False)

        for param in self.w2v.parameters():
            param.requires_grad = False
        print("Finish loading pretrained Wav2Vec...")
        

        self.weights = nn.Parameter(torch.full((12,),1.0))


        self.dropout = nn.Dropout(0.1)
        self.decoder_type = decoder_type
        # if decoder_type == 'linear':
        #     self.decoder_time = nn.Linear(wavelength//4, 1)
        #     self.decoder_dim = nn.Linear(768, 1)
        if decoder_type == 'cnn':
            self.cnn_1 = nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding='same'),
                nn.BatchNorm1d(256),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_2 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=5, padding='same'),
                nn.BatchNorm1d(256),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_3 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=5,padding='same'),
                nn.BatchNorm1d(256),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.cnn_4 = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=11),
                nn.BatchNorm1d(256),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            self.flatten = nn.Flatten()
            self.out = nn.Linear(10496,1)

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
        elif decoder_type == 'CNN_Linear_lightweight':
            self.conv = nn.Sequential(nn.Conv1d(128, 64, kernel_size=7, padding='same'),
                                   nn.BatchNorm1d(64),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),
                                   nn.Conv1d(64, 32, kernel_size=5, padding='same'),
                                   nn.BatchNorm1d(32),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),
                                   nn.Conv1d(32, 16, kernel_size=3, padding='same'),
                                   nn.BatchNorm1d(16),
                                   nn.MaxPool1d(4),
                                   nn.ReLU(),
                                   nn.Dropout(0,1),)
        
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(368, 1)
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
        # print("123333333333333333333333333333333333333333333333333333")
        # rep = self.w2v(wave)

        weighted_sum = 'y'
        if weighted_sum == 'y':
            weights = F.softmax(self.weights,dim=0)
            wei = 0
            ws = torch.zeros_like(rep[0][0])
            ws = torch.permute(ws,(1,0,2))
            for i in rep:
                temp = torch.permute(i[0],(1,0,2))
                temp = temp * weights[wei]
                wei = wei + 1
                ws = ws + temp
            rep = ws

        if self.decoder_type == 'linear':
            rep_time_reduction = self.decoder_time(rep.permute(0,2,1)).permute(0,2,1)
            out = self.decoder_dim(rep_time_reduction)
        elif self.decoder_type == 'cnn':
            x = self.cnn_1(rep.permute(0,2,1))
            x = self.cnn_2(x)
            x = self.cnn_3(x)
            x = self.cnn_4(x)
            out = self.flatten(x)
            out = self.out(out)

        elif self.decoder_type == 'CNN_Linear':
            cnn_out = self.conv(rep.permute(0,2,1))
            out = self.flatten(cnn_out)
            out = self.out(out)
        elif self.decoder_type == 'CNN_Linear_lightweight':
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
