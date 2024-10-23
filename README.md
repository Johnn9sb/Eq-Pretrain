![](https://img.shields.io/static/v1?label=python&message=>3.8&color=yellow)
# Earthquake-Pretraining
PyTorch implementation of "Earthquake Waveform Pre-trained Model" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Introduction
Earthquake monitoring and warning are critical for disaster early warning systems. While deep learning has been extensively studied in this field, the application of **self-supervised learning** remains relatively unexplored. Self-supervised learning can learn useful representations without requiring extensive labeled data, and models trained in this way can be fine-tuned for specific tasks with better efficiency.

## Proposed Model
We propose a **Large Earthquake Model (LEM)** to evaluate the effectiveness of self-supervised learning in earthquake-related tasks, focusing on **p-phase picking** and **magnitude estimation**.
![LEM](/docs/model.png)

## Key Results
Our experiments show that LEM delivers competitive results compared to state-of-the-art models, achieving satisfactory performance using only **0.1% of the labeled data**. This approach offers a potential solution to reduce the time and effort required for seismic data labeling.

## Pre-trained models
You need to download the pre-trained weights and input the savepath of the weights into the required model to perform fine-tuning.

(New)Download Link: [download](https://drive.google.com/file/d/1sXjPTJ5Y8bNmJERgkAUZx7BLOTsrbeFP/view?usp=sharing)

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

