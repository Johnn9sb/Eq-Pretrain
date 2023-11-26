# Earthquake-Pretraining
PyTorch implementation of "Earthquake Waveform Pre-trained Model" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Requirements and Installation
+ [PyTorch](https://pytorch.org/) version >= 1.10.0
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# If you have PyTorch installed, please ignore this line.
```
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
python load_model.py --config-dir ./config --config-name eq_pretrain
```
Please run and edit your scripts within `load_model.py`.
This is necessary to correctly load models in the fairseq format.
