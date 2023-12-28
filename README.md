![](https://img.shields.io/static/v1?label=python&message=>3.8&color=yellow)
# Earthquake-Pretraining
PyTorch implementation of "Earthquake Waveform Pre-trained Model" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Pre-trained models
You need to download the pre-trained weights and input the savepath of the weights into the required model to perform fine-tuning.

(CNN*2 Mask=5)Download Link: [download](https://drive.google.com/file/d/1sXjPTJ5Y8bNmJERgkAUZx7BLOTsrbeFP/view?usp=sharing)

(Old)Download Link: [download](https://drive.google.com/file/d/1QRpMPg4Q-gOQpfDoS5NbmiVzIMb6njS9/view?usp=sharing)

## Requirements and Installation
```
git clone https://github.com/Johnn9sb/Eq-Pretrain.git
```
+ [PyTorch](https://pytorch.org/) version >= 1.10.0
+ Python version >= 3.8
+ To install and develop this implementation locally:
```
cd Eq-Pretrain
pip install -r requirements.txt
pip install --editable ./model/fairseq/
```
+ If there is a need to test whether the environment installation is successful, the corresponding method can be found in the `load_model.py` file.

