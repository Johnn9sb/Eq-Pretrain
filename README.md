# Earthquake-Pretraining
Earthquake Waveform Pre-trained Model
PyTorch implementation of "Earthquake Waveform Pre-trained Model" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Requirements and Installation
+ [PyTorch](https://pytorch.org/) version >= 1.10.0
+ Python version >= 3.8
+ To avoid environment issues, please run our script in the Anaconda environment.
+ To install and develop this implementation locally:
```
git clone https://github.com/j0hnng/Eq-Pretrain.git
mv Eq-Pretrain fairseq
cd fairseq
pip install --editable ./
```
+ How to load models in this implementation
```
python ./fairseq_cli/load_model.py --config-dir ./config --config-name eq_pretrain
```
Please run and edit your scripts within `load_model.py` in `./fairseq_cli`.
This is necessary to correctly load models in the fairseq format.
