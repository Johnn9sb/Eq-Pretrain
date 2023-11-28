import torch

# Load model with this command!!!
from model.wav2vec2 import Wav2Vec2Model,Wav2Vec2Config
w2v = Wav2Vec2Model(Wav2Vec2Config)

checkpoint = torch.load('checkpoint.pt')
w2v.load_state_dict(checkpoint['model'], strict=True)

print(w2v)
print('!!!!!!!!!!!!!!!!!!!!!!')