# Magnitude Estimaion

## Install Dependency
```shell=
$ cd ../seisbench

# 更改 seisbench/__init__.py 內的 "cache_root"，用來放資料集 cache (需要大容量的路徑)

$ pip install .
```

## Fine-tuning
* 重要指令不能為空
    * ```--save_path```: checkpoint 存放位置 (./result/<save_path>)
    * ```--model_opt```: 模型種類 (ex. w2v, magnet...)
    * ```--decoder_type```: For w2v, 指定的 decoder 類別 (ex. linear, CNN_Linear, CNN_LSTM)

* 建議下的指令
    * ```--batch_size```, ```--epochs```, ```--lr```
    * ```--workers```: Dataloader 使用的執行緒數量
    * ```--level```: CWA dataset 篩選，通常設為 "4"
    * ```--without_noise```: 將雜訊資料捨棄 (True/False)
    * ```--aug```: 做 data augmentation (ex. Gaussian noise...)
    * ```--w2v_path```: 如果不要 load pretrained，請給 "None" 值

* 篩選資料集指令
    * ```--instrument```: 指定儀器 channel (ex. HL, EH, HH)
    * ```--location```: 指定儀器 location (ex. 10, 0, 20)
    * ```--epidis```: 指定波型的 epicentral distance <= 某值
    * ```--snr```: 指定波型的 SNR >= 某值

```shell=
$ CUDA_VISIBLE_DEVICES=<gpu_id> python train.py \
    --save_path <save_path> \
    --model_opt <model_opt> \
    --decoder_type <decoder_type> \
```

## Testing
* 指令規則如上
* 測試專屬指令
    * ```--p_timestep```: 測試時固定 P-phase 在某時間點

```shell=
$ CUDA_VISIBLE_DEVICES=<gpu_id> python test.py \
    --save_path <save_path> \
    --model_opt <model_opt> \
    --decoder_type <decoder_type> \
    --p_timestep <p_timestep> \
```

## Output
* 結果存於 ./result/<save_path>/score
