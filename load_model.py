import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Load model with this command!!!
# from model.wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
from model.ws_wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
w2v = Wav2Vec2Model(Wav2Vec2Config)

# checkpoint = torch.load('./pretrain/pretrainmask5/pretrainmask5.pt')
# w2v.load_state_dict(checkpoint['model'], strict=True)

print(w2v)
print('!!!!!!!!!!!!!!!!!!!!!!')
random_tensor = torch.rand(5, 3, 3000)
a = w2v(random_tensor)
print(a)
sys.exit()

# init
weights = nn.Parameter(torch.full((12,),1.0))
print(weights)
print(torch.sum(weights))

# forward
weights = F.softmax(weights,dim=0)
print(weights)
print(torch.sum(weights))

wei = 0
ws = torch.zeros_like(a[0][0])
ws = torch.permute(ws,(1,0,2))
for i in a:
    print(type(i[0]))
    print(len(i[0]))
    print(i[0].shape)
    x = torch.permute(i[0],(1,0,2))
    x = x * weights[wei]
    wei = wei + 1
    ws = ws + x
print(ws.shape)

sys.exit()

print(len(a))
print(type(a))
print(len(a[0]))
print(type(a[0]))
print(a[0][0].shape)
print(type(a[0][0]))




