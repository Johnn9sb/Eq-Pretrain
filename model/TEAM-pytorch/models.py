import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from scipy.stats import norm
import json


def my_collate(batch):
    return batch


class MLP(nn.Module): 
    def __init__(self, input_shape, dims):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_features=input_shape[0], out_features=dims[0], bias=True))
        for i in range(len(dims)-1):
            self.linears.append(nn.Linear(in_features=dims[i], out_features=dims[i+1], bias=True))

        for module in self.linears.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

        
    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        return x


class MixtureOutput(nn.Module):
    def __init__(self, input_shape, n, d=1, eps=1e-4, bias_mu=1.8, bias_sigma=0.2, name=None):
        super(MixtureOutput, self).__init__()
        self.input_shape = input_shape
        self.eps = eps
        self.n = n
        self.d = d

        self.alpha_linear = nn.Linear(in_features=input_shape[-1], out_features=n)
        self.mu_linear = nn.Linear(in_features=input_shape[-1], out_features=n*d)
        self.sigma_linear = nn.Linear(in_features=input_shape[-1], out_features=n*d)

        for i in [self.alpha_linear, self.mu_linear, self.sigma_linear]:
            for module in i.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight.data)
                    if module.bias is not None:
                        nn.init.constant_(module.bias.data, 0)
                        if i == self.mu_linear:
                            nn.init.constant_(module.bias.data, bias_mu)
                        if i == self.sigma_linear:
                            nn.init.constant_(module.bias.data, bias_sigma)
            
    def forward(self, x):
        alpha = F.softmax(self.alpha_linear(x), dim=-1)
        alpha = torch.reshape(alpha, (-1, self.n, 1))

        mu = self.mu_linear(x) 
        mu = torch.reshape(mu, (-1, self.n, self.d))

        sigma = F.relu(self.sigma_linear(x)) + self.eps  # Add epsilon to avoid division by 0
        sigma = torch.reshape(sigma, (-1, self.n, self.d))

        out = torch.cat((alpha, mu, sigma),2)
        return out


class NormalizedScaleEmbedding(nn.Module):
    def __init__(self, input_shape, mlp_dims, downsample=1, eps=1e-8):
        super(NormalizedScaleEmbedding, self).__init__()
        self.eps=eps
        self.inp_shape = input_shape
        self.Conv2D_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(downsample, 1), stride=(downsample, 1))
        self.Conv2D_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(16, 3), stride=(1, 3))

        if input_shape[-1]==3:
            self.Conv1D_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        if input_shape[-1]==6:
            self.Conv1D_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16)
        self.Conv1D_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16)
        self.Conv1D_3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=8)
        self.Conv1D_4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.Conv1D_5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4)

        self.pool1D_1= nn.MaxPool1d(kernel_size=2)
        self.pool1D_2= nn.MaxPool1d(kernel_size=2)
        self.pool1D_3= nn.MaxPool1d(kernel_size=2)

        self.MLP = MLP(input_shape=(865,), dims=mlp_dims)

        for i in [self.Conv2D_1, self.Conv2D_2, self.Conv1D_1, self.Conv1D_2, self.Conv1D_3, self.Conv1D_4, self.Conv1D_5]:
            for module in i.modules():
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
        
    def forward(self, x):  #(250, 3000, 6)
        scale = torch.log(torch.amax(input=torch.abs(x), dim=(1,2)) + self.eps) /100  #(250,)
        scale = torch.unsqueeze(scale,1) #(250, 1)
        x = x / (torch.amax(input=torch.abs(x), dim=(1,2), keepdims=True) + self.eps)  #(250, 3000, 6)
        x = torch.unsqueeze(x,1)  #(250, 1, 3000, 6) 
        x = F.relu(self.Conv2D_1(x))  #(250, 8, 600, 6)
        x = F.relu(self.Conv2D_2(x))  #(250, 32, 585, 2)
        x = torch.reshape(x, (x.shape[0], 32 * self.inp_shape[-1] // 3, -1))  #(250, 64, 585)
        x = self.pool1D_1(F.relu(self.Conv1D_1(x)))  #(250, 64, 285)
        x = self.pool1D_2(F.relu(self.Conv1D_2(x)))  #(250, 128, 135)
        x = self.pool1D_3(F.relu(self.Conv1D_3(x)))  #(250, 32, 64)
        x = F.relu(self.Conv1D_4(x))  #(250, 32, 57)
        x = F.relu(self.Conv1D_5(x))  #(250, 16, 54)
        x = torch.flatten(x, 1)  #(250, 864)
        x = torch.cat((x, scale), -1)  #(250, 865)
        x = self.MLP(x)  #(250, 500)
        return x


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.shape) == 4:
            tmp = x.contiguous().view(-1, x.size(-2), x.size(-1))  
        elif len(x.shape) == 3:
            tmp = x.contiguous().view(-1, x.size(-1))
        else: print('TimeDistributed dimension do not support')

        y = self.module(tmp)
        if len(y.shape) == 3:
            y = y.contiguous().view(x.size(0), x.size(1), -1, y.size(-1))
        elif len(y.shape) == 2:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) 
        else: print('TimeDistributed dimension do not support')

        return y


class LayerNorm(nn.Module):
    def __init__(self, input_shape, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(input_shape[-1:]))
        self.gamma = nn.Parameter(torch.ones(input_shape[-1:]))

    def forward(self, x):
        m = x.mean(-1, keepdims=True)
        s = torch.mean(torch.square(x-m), dim=-1, keepdims=True)
        z = (x - m) / torch.sqrt(s + self.eps)
        output = self.gamma*z + self.beta

        return output


class PositionEmbedding(nn.Module):
    def __init__(self, device, wavelength, emb_dim, borehole=False, rotation=None, rotation_anchor=None,**kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.wavelength = wavelength  # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            c, s = np.cos(rotation), np.sin(rotation)
            self.rotation_matrix = torch.Tensor(np.array(((c, -s), (s, c)), dtype=float)).to(device)
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelength[0]
        min_lon, max_lon = wavelength[1]
        min_depth, max_depth = wavelength[2]
        assert emb_dim % 10 == 0
        if borehole:
            assert emb_dim % 20 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        if borehole:
            depth_dim = emb_dim // 20
        self.lat_coeff = (2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))).astype('float32')
        self.lon_coeff = (2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))).astype('float32')
        self.depth_coeff = (2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))).astype('float32')
        self.lat_coeff = torch.from_numpy(self.lat_coeff).to(device)
        self.lon_coeff = torch.from_numpy(self.lon_coeff).to(device)
        self.depth_coeff = torch.from_numpy(self.depth_coeff).to(device)

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        self.mask = np.zeros(emb_dim) 
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim = depth_dim * 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.mask = torch.unsqueeze(torch.unsqueeze(torch.LongTensor(self.mask.astype('int64')),0),0).to(device)
        
    def forward(self, x):
        fake_borehole = False

        if x.shape[-1] == 3:
            fake_borehole = True

        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1]
            lon_base = lon_base * torch.cos(lat_base * np.pi / 180)

            lat_base = lat_base - self.rotation_anchor[0] 
            lon_base = lon_base - self.rotation_anchor[1] * np.cos(self.rotation_anchor[0] * np.pi / 180) 
            
            latlon = torch.stack([lat_base, lon_base], axis=-1)
            rotated = latlon @ self.rotation_matrix

            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff
            lon_base = x[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        if self.borehole:
            if fake_borehole:
                depth_base = x[:, :, 2:3] * self.depth_coeff * 0
                depth2_base = x[:, :, 2:3] * self.depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * self.depth_coeff
            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base),
                                torch.sin(depth2_base), torch.cos(depth2_base)], -1) 
        else:
            output = torch.cat([torch.sin(lat_base), torch.cos(lat_base), 
                                torch.sin(lon_base), torch.cos(lon_base),
                                torch.sin(depth_base), torch.cos(depth_base)], -1)
        
        mask = self.mask.expand(x.shape[0], x.shape[1], 500)
        output = torch.gather(output, -1, mask)
        return output

        
class WaveformsEmbedding(nn.Module):  
    def __init__(self, input_shape, downsample, mlp_dims):
        super(WaveformsEmbedding, self).__init__()
        self.NormalizedScaleEmbedding = NormalizedScaleEmbedding(input_shape=input_shape, downsample=downsample, mlp_dims=mlp_dims)
        self.TimeDistributed = TimeDistributed(self.NormalizedScaleEmbedding)
        self.LayerNorm = LayerNorm(mlp_dims)

    def forward(self, x, waveforms_mask):
        waveforms_mask = torch.unsqueeze(waveforms_mask,-1)
        x = x * waveforms_mask
        x = self.TimeDistributed(x)
        x = self.LayerNorm(x)
        return x


class TotalEmbedding(nn.Module):
    def __init__(self, input_shape, downsample, mlp_dims=(500,500,500), wavelength=[[0.1,25],[0.1,25],[0.01,10]], device='cpu',
                 borehole=False, rotation=None, rotation_anchor=None, alternative_coords_embedding=False, **kwargs):
        super(TotalEmbedding, self).__init__()
        self.device = device
        self.alternative_coords_embedding = alternative_coords_embedding
        self.Input_Station_emb = PositionEmbedding(wavelength=wavelength, emb_dim=mlp_dims[-1], borehole=borehole, rotation=rotation, rotation_anchor=rotation_anchor, device=device)
        self.Target_Station_emb = PositionEmbedding(wavelength=wavelength, emb_dim=mlp_dims[-1], borehole=borehole, rotation=rotation, rotation_anchor=rotation_anchor, device=device)
        self.WaveformsEmbedding = WaveformsEmbedding(input_shape=input_shape, downsample=downsample, mlp_dims=mlp_dims)

    def forward(self, waveforms, inputs, targets):
        waveforms_mask = torch.unsqueeze(torch.any(torch.any(torch.not_equal(waveforms, 0),-1),-1),-1).to(self.device)
        inputs_mask = torch.unsqueeze(torch.any(torch.not_equal(inputs,0),-1),-1).to(self.device)
        targets_mask = torch.unsqueeze(torch.any(torch.not_equal(targets,0),-1),-1).to(self.device)

        tmp = np.concatenate([np.ones((inputs.shape[0],inputs.shape[1])), np.zeros((inputs.shape[0],targets.shape[1]))], axis=1)
        att_mask = torch.Tensor(tmp).to(self.device)

        waveforms = self.WaveformsEmbedding(waveforms, waveforms_mask)
        if not self.alternative_coords_embedding:
            inputs = self.Input_Station_emb(inputs)
            inputs = inputs * inputs_mask
            outputs = waveforms + inputs
        else:
            outputs = torch.cat((waveforms, inputs),-1)

        targets = self.Target_Station_emb(targets)
        targets = targets * targets_mask

        outputs = torch.cat((outputs, targets), 1)
        input_station_mask = torch.logical_or(waveforms_mask, inputs_mask)
        station_mask = torch.cat((input_station_mask, targets_mask),1).to(self.device)
        return outputs, att_mask, station_mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, total_station, n_heads, mlp_dims, att_dropout, initializer_range, infinity=1e6, device='cpu', **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.att_dropout = att_dropout
        self.n_heads = n_heads
        self.infinity = infinity
        self.d_model = mlp_dims[-1]
        self.stations = total_station
        self.d_key = self.d_model // n_heads
        assert self.d_model % n_heads == 0
        
        initialization_1 = torch.torch.distributions.Uniform(low=-initializer_range, high=initializer_range).sample((self.d_model, self.d_key*n_heads))
        initialization_2 = torch.torch.distributions.Uniform(low=-initializer_range, high=initializer_range).sample((self.d_key*n_heads, self.d_model))
        self.WQ = nn.Parameter(initialization_1)
        self.WK = nn.Parameter(initialization_1)
        self.WV = nn.Parameter(initialization_1)
        self.WO = nn.Parameter(initialization_2)

    def forward(self, x, att_mask, station_mask):
        d_key = self.d_key  
        n_heads = self.n_heads 
        q = torch.matmul(x, self.WQ) 
        q = torch.reshape(q, (-1, self.stations, d_key, n_heads)) 
       
        q = q.permute((0, 3, 1, 2))  # (batch, n_heads, stations, key)
        k = torch.matmul(x, self.WK)  # (batch, stations, key*n_heads)
        k = torch.reshape(k, (-1, self.stations, d_key, n_heads))
        k = k.permute((0, 3, 2, 1))  # (batch, n_heads, key, stations)
        score = torch.matmul(q, k) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)

        inv_mask = torch.unsqueeze(torch.logical_not(station_mask),-1)  # (batch, stations, 1, 1)
        mask_B = inv_mask.permute((0, 2, 3, 1))  # (batch, 1, 1, stations)
        score = score - mask_B * self.infinity
        
        inv_mask = torch.unsqueeze(torch.unsqueeze(torch.logical_not(att_mask),-1),-1)  # (batch, stations, 1, 1)
        mask_B = inv_mask.permute((0, 2, 3, 1))  # (batch, 1, 1, stations)
        score = score - mask_B * self.infinity

        score = F.softmax(score, dim=-1)
        if self.att_dropout > 0:
            dropout = nn.Dropout(p=self.att_dropout)
            score = dropout(score)
        v = torch.matmul(x, self.WV)  # (batch, stations, key*n_heads)
        v = torch.reshape(v, (-1, self.stations, d_key, n_heads))
        v = v.permute((0, 3, 1, 2))  # (batch, n_heads, stations, key)

        o = torch.matmul(score, v)  # (batch, n_heads, stations, key)
        o = o.permute((0, 2, 1, 3))  # (batch, stations, n_heads, key)
        o = torch.reshape(o, (-1, self.stations, n_heads * d_key))
        o = torch.matmul(o, self.WO)
        o = torch.abs(o * station_mask)
        return o


class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, device, **kwargs):
        super(PointwiseFeedForward, self).__init__()
        tmp1 = torch.nn.init.xavier_uniform_(torch.empty(emb_dim, hidden_dim), gain=1).to(device)
        self.kernel1 = nn.Parameter(tmp1)
        tmp2 = torch.nn.init.xavier_uniform_(torch.empty(hidden_dim, emb_dim), gain=1).to(device)
        self.kernel2 = nn.Parameter(tmp2)

        self.bias1 = nn.Parameter(torch.zeros(hidden_dim,))
        self.bias2 = nn.Parameter(torch.zeros(emb_dim,))

    def forward(self, x, station_mask):
        x = F.gelu(torch.matmul(x, self.kernel1) + self.bias1)
        x = torch.matmul(x, self.kernel2) + self.bias2
        x = x * station_mask
        return x


class TransformerLayer(nn.Module):
    def __init__(self, total_station, hidden_dropout=0.0,  mlp_dims=(500,500,500), device='cpu', 
                 mad_params={}, ffn_params={}, norm_params={}):
        super(TransformerLayer, self).__init__()
        self.hidden_dropout = hidden_dropout
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(total_station=total_station, mlp_dims=mlp_dims, device=device, **mad_params)
        
        self.PointwiseFeedForward = PointwiseFeedForward(emb_dim=mlp_dims[-1], device=device, **ffn_params)
        self.LayerNorm1 = LayerNorm(input_shape=mlp_dims, **norm_params)
        self.LayerNorm2 = LayerNorm(input_shape=mlp_dims, **norm_params)

    def forward(self, x, att_mask, station_mask):
        modified_x= self.MultiHeadSelfAttention(x, att_mask,station_mask)
        if self.hidden_dropout > 0:
            modified_x = nn.Dropout(self.hidden_dropout)(modified_x)
        x = self.LayerNorm1(x + modified_x)

        modified_x = self.PointwiseFeedForward(x, station_mask)
        if self.hidden_dropout > 0:
            modified_x = nn.Dropout(self.hidden_dropout)(modified_x)
        x = self.LayerNorm2(x + modified_x)
        return x


class Transformer(nn.Module):
    def __init__(self, device, total_station=45, layers=6, hidden_dropout=0.0,  mlp_dims=(500,500,500),
                 mad_params={}, ffn_params={}, norm_params={}):
        super(Transformer, self).__init__()
        tmp = TransformerLayer(total_station=total_station, hidden_dropout=hidden_dropout, device=device,
                                mlp_dims=mlp_dims, mad_params=mad_params, ffn_params=ffn_params, norm_params=norm_params)
        self.layers = nn.ModuleList([copy.deepcopy(tmp) for i in range(layers)])

    def forward(self, x, att_mask, station_mask):
        for layer in self.layers:
            x = layer(x, att_mask, station_mask)
        return x


class To_GaussianDistribution(nn.Module):
    def __init__(self, n_pga_targets, emb_dim=500, output_mlp_dims=(150, 100, 50, 30, 10), pga_mixture=5):
        super(To_GaussianDistribution, self).__init__()
        self.n_pga_targets = n_pga_targets
        self.MLP = MLP((emb_dim,), output_mlp_dims)
        self.TimeDistributed1 = TimeDistributed(self.MLP)

        self.MixtureOutput = MixtureOutput((output_mlp_dims[-1],), pga_mixture, bias_mu=-5, bias_sigma=1)
        self.TimeDistributed2 = TimeDistributed(self.MixtureOutput)

    def forward(self, x):
        x = x[:, -self.n_pga_targets:, :]
        x = self.TimeDistributed1(x)
        x = self.TimeDistributed2(x)
        return x


class FullModel(nn.Module):
    def __init__(self, device, input_shape, downsample, mlp_dims, wavelength, total_station,
                 layers, hidden_dropout, n_pga_targets, output_mlp_dims, pga_mixture,
                 borehole=None,rotation=None, rotation_anchor=None,alternative_coords_embedding=False,
                 mad_params={}, ffn_params={},model_type="transformer"):
        super(FullModel, self).__init__()
        self.TotalEmbedding = TotalEmbedding(input_shape=input_shape, downsample=downsample, mlp_dims=mlp_dims, device=device,
                                             wavelength=wavelength, borehole=borehole, rotation=rotation, 
                                             rotation_anchor=rotation_anchor, alternative_coords_embedding=alternative_coords_embedding)
        print(f"model_type:{model_type}")
        if(model_type=="transformer"):
            self.Transformer = Transformer(total_station=total_station, device=device, layers=layers, hidden_dropout=hidden_dropout, mlp_dims=mlp_dims,
                                        mad_params=mad_params, ffn_params=ffn_params)
        else:
            self.Transformer = Conformer(total_station=total_station, device=device, layers=layers, hidden_dropout=hidden_dropout, mlp_dims=mlp_dims,
                                        mad_params=mad_params, ffn_params=ffn_params)

        self.To_GaussianDistribution = To_GaussianDistribution(n_pga_targets=n_pga_targets, emb_dim=mlp_dims[-1], 
                                                               output_mlp_dims=output_mlp_dims, pga_mixture=pga_mixture)

    def forward(self, waveforms, inputs, targets):
        x, att_mask, station_mask = self.TotalEmbedding(waveforms, inputs, targets)
        x = self.Transformer(x, att_mask, station_mask)
        x = self.To_GaussianDistribution(x)
        return x

def build_transformer_model(max_stations,
                            waveform_model_dims=(500, 500, 500),
                            output_mlp_dims=(150, 100, 50, 30, 10),
                            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                            mad_params={"n_heads": 10,
                                        "att_dropout": 0.0,
                                        "initializer_range": 0.02
                                        },
                            ffn_params={'hidden_dim': 1000},
                            transformer_layers=6,
                            hidden_dropout=0.0,
                            n_pga_targets=0,
                            pga_mixture=5,
                            borehole=False,
                            trace_length=3000,
                            downsample=5,
                            rotation=None,
                            rotation_anchor=None,
                            alternative_coords_embedding=False,
                            device='cpu',
                            model="transformer",
                            **kwargs):
    if kwargs:
            print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')
    emb_dim = waveform_model_dims[-1]
    mad_params = mad_params.copy()  
    ffn_params = ffn_params.copy()

    if borehole:
        input_shape = (trace_length, 6)
        metadata_shape = (4,)
    else:
        input_shape = (trace_length, 3)
        metadata_shape = (3,)

    mad_params = mad_params.copy() 
    ffn_params = ffn_params.copy()
    print('n_pga_targets',n_pga_targets)
    total_station = max_stations + n_pga_targets
    parameters = {'input_shape':input_shape, 'downsample':downsample, 'mlp_dims':waveform_model_dims,
                  'wavelength':wavelength, 'total_station':total_station, 'layers':transformer_layers,
                  'hidden_dropout':hidden_dropout, 'n_pga_targets':n_pga_targets, 'output_mlp_dims':output_mlp_dims,
                  'pga_mixture':pga_mixture, 'borehole':borehole, 'rotation':rotation, 'rotation_anchor':rotation_anchor,
                  'alternative_coords_embedding':alternative_coords_embedding, 
                  'mad_params':mad_params, 'ffn_params':ffn_params, 'device':device,'model_type':model,
                 }
    return FullModel(**parameters)


def mixture_density_loss(y_true, y_pred, device='cpu',  eps=1e-6, d=1, mean=True, print_shapes=False):

    alpha = y_pred[:, :, 0]  
    density = torch.ones(y_pred.shape[0], y_pred.shape[1]).to(device) 
    for j in range(d): 
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = torch.max(sigma, (torch.ones(sigma.shape)*eps).to(device))
        density = density * 1 / (np.sqrt(2 * np.pi).astype('float32') * sigma) * torch.exp(-(y_true[:, j] - mu) ** 2 / (2 * sigma ** 2))
    density = density * alpha
    density = torch.sum(density, axis=-1)
    density = density + eps
    loss = - torch.log(density)  

    if mean:
        return torch.mean(loss)
    else:
        return loss 

def time_distributed_loss(y_true, y_pred, loss_func, device='cpu', norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = y_true.contiguous().view(-1, (y_pred.shape[-1] - 1) // 2, 1)     #(batch, 20, 5, 3) -> (batch*20, 5, 3)
    y_pred = y_pred.contiguous().view(-1, y_pred.shape[-2], y_pred.shape[-1]) #(batch, 20, 1, 1) -> (batch*20, 1, 1)
    loss = loss_func(y_true, y_pred, device, **kwloss) 
    
    if mean: 
        return torch.mean(loss)
    loss = loss / norm
    if summation:
        loss = torch.sum(loss)

    return loss


class EnsembleEvaluateModel:
    def __init__(self, config, experiment_path, max_ensemble_size=None, loss_limit=None, batch_size=64, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.config = config
        self.ensemble = config.get('ensemble', 1)
        true_ensemble_size = self.ensemble
        if max_ensemble_size is not None:
            self.ensemble = min(self.ensemble, max_ensemble_size)
        self.models = []
        filter_borehole = json.load(open('dataset_config/idx_st_loc_filter.json'))
        self.filter_borehole = [int(key) for key, value in filter_borehole.items() if value == 1]
        
        print('loading weights......')
        self.ensemble = 1
        for ens_id in range(self.ensemble):
            ens_id = 0
            weight_path = os.path.join(experiment_path, str(ens_id))
            weight_file = sorted([x for x in os.listdir(weight_path) if x[:11] == 'checkpoint_'])[-1] 
            weight_file = os.path.join(experiment_path, str(ens_id),weight_file)
            self.weight_file = weight_file
            model_params = config['model_params'].copy()    
            print('weight_file',weight_file)

            if config['training_params'].get('ensemble_rotation', False):
                model_params['rotation'] = np.pi / 4 * ens_id / (true_ensemble_size - 1) 

            tmp_model = build_transformer_model(device=device, **model_params).to(device)
            tmp_model.load_state_dict(torch.load(weight_file)['model_weights'])
            
            self.models = self.models + [tmp_model]
        self.loss_limit = loss_limit

    def predict_generator(self, generator, **kwargs):
        preds = torch.tensor([], device=self.device) 
        for i, model in enumerate(self.models):
            dataloader = DataLoader(generator, batch_size=None, collate_fn=my_collate,num_workers=0)
            tmp_preds = torch.tensor([], device=self.device) 

            model.eval()
            with torch.no_grad():
                for data_idx, data in enumerate(dataloader): 
                    waveforms = data[0][0].to(self.device)
                    inputs = data[0][1].to(self.device)
                    targets = data[0][2].to(self.device)
                    
                    outputs = model(waveforms, inputs, targets) 
                    labels = data[1][0].to(self.device)

                    outputs = outputs.squeeze(0)
                    outputs = outputs[self.filter_borehole].unsqueeze(0)
                    tmp_preds = torch.cat((tmp_preds, outputs), 0)

            tmp_preds = torch.unsqueeze(tmp_preds, 0)
            preds = torch.cat((preds, tmp_preds), 0) 
        preds = preds.cpu().numpy()
        return self.merge_preds(preds)

    @staticmethod
    def merge_preds(preds):
        pred_item = np.concatenate(preds, axis=-2) 
        pred_item[:, :, :, 0] /= np.sum(pred_item[:, :, :, 0], axis=-1, keepdims=True) 

        return pred_item

    def load_weights(self, weights_path):
        tmp_models = self.models
        self.models = []
        removed_models = 0
        for ens_id, model in enumerate(tmp_models):
            if self.loss_limit is not None:
                hist_path = os.path.join(weights_path, f'{ens_id}', 'hist.pkl')
                with open(hist_path, 'rb') as f:
                    hist = pickle.load(f)
                if np.min(hist['val_loss']) > self.loss_limit:
                    removed_models = removed_models + 1
                    continue

            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')
            weight_file = sorted([x for x in os.listdir(tmp_weights_path) if x[:5] == 'event'])[-1]
            weight_file = os.path.join(tmp_weights_path, weight_file)
            model.load_weights(weight_file)
            self.models = self.models + [model]

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')
    
    def prediction_generator_partial_station(self, generator, **kwargs):
        end_list = []
        end_label_list = []

        thresholds = np.array([0.081, 0.25, 0.81, 1.4, 2.5, 4.4, 8.0])
        for i, model in enumerate(self.models):
            dataloader = DataLoader(generator, batch_size=None, collate_fn=my_collate,num_workers=0)
            output_list = []
            output_label_list = []

            model.eval()
            with torch.no_grad():
                for data_idx, data in enumerate(dataloader): 
                    waveforms = data[0][0].to(self.device)
                    inputs = data[0][1].to(self.device)
                    targets = data[0][2].to(self.device)
                    outputs = model(waveforms, inputs, targets)   #(batch, 20, 5, 3)
                    labels = data[1][0].to(self.device)
                    
                    outputs = outputs.squeeze(0)
                    outputs = outputs[self.filter_borehole]
                    labels = labels[self.filter_borehole]
                    selected_indices = torch.nonzero(labels!=0).squeeze(dim=-1).to(self.device).long()

                    outputs = outputs[selected_indices]
                    labels = labels[selected_indices]
                    
                    outputs_select = outputs < 1  #(249,5,3)
                    outputs_select[:,:,0] = False
                    outputs_select[:,:,2] = False
                    tmp_outputs = torch.pow(10, outputs) / 10
                    outputs[outputs_select] = tmp_outputs[outputs_select]
                    labels = torch.log10(labels*10)
                    output_list.append(outputs)
                    output_label_list.append(labels)

            output_list = torch.concat(output_list,0)
            output_label_list = torch.concat(output_label_list,0)
            end_list.append(output_list)
            end_label_list.append(output_label_list)
        end_list = torch.concat(end_list,0)
        end_label_list = torch.concat(end_label_list,0)
        end_list = end_list.cpu().numpy()
        end_label_list = end_label_list.cpu().numpy()
        return end_list, end_label_list
