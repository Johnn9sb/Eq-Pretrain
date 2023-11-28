![](https://img.shields.io/static/v1?label=python&message=>3.8&color=yellow)
# Earthquake-Pretraining
PyTorch implementation of "Earthquake Waveform Pre-trained Model" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Pre-trained models
Download Link: [download](https://drive.google.com/file/d/1QRpMPg4Q-gOQpfDoS5NbmiVzIMb6njS9/view?usp=sharing)


## Requirements and Installation
+ [PyTorch](https://pytorch.org/) version >= 1.10.0
```
pip install -r requirements.txt
```
+ Python version >= 3.8
+ To install and develop this implementation locally:
```
git clone https://github.com/Johnn9sb/Eq-Pretrain.git
cd Eq-Pretrain
pip install --editable ./model/fairseq/
```
+ If there is a need to load the model, the corresponding method can be found in the `load_model.py` file.

