from model import Wav2Vec_Mag
import torch
import torch.nn.functional as F


w2v = Wav2Vec_Mag(
    decoder_type='cnn', 
    wavelength=3000, 
    checkpoint_path='None',
    device='cpu'
)

checkpoint = torch.load('/mnt/nas3/johnn9/mag_checkpoint/data2vec_wei/checkpoint_last.pt')
w2v.load_state_dict(checkpoint['model'], strict=True)

print("Model weights:", w2v.weights)
weights = F.softmax(w2v.weights,dim=0)
print("Softmax weights:", weights)