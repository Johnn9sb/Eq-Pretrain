import torch

# Load model with this command!!!
from model.ds_wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
w2v = Wav2Vec2Model(Wav2Vec2Config)

# checkpoint = torch.load('./pretrain/pretrainmask5/pretrainmask5.pt')
# w2v.load_state_dict(checkpoint['model'], strict=True)

print(w2v)
print('!!!!!!!!!!!!!!!!!!!!!!')
random_tensor = torch.rand(1, 3, 3000)
a = w2v(random_tensor)
print(a)
