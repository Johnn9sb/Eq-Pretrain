# from model.EQ_Pretrain.load_model import load_fairseq
# if __name__ == "__main__":
#     model_w2v = load_fairseq('./model/EQ_Pretrain/config',"eq_pretrain")

import torch
from model.wav2vec2 import Wav2Vec2Model,Wav2Vec2Config



w2v = Wav2Vec2Model(Wav2Vec2Config)
print(w2v)
checkpoint = torch.load('checkpoint.pt')
# print(checkpoint.keys())
# print(checkpoint['model'].keys())
w2v.load_state_dict(checkpoint['model'], strict=True)

print(w2v)
# Load Pre-trained Weights
# model.load_state_dict(Checkpoint["model"], strict=True)

print('!!!!!!!!!!!!!!!!!!!!!!')