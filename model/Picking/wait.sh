#!/bin/bash

while true; do
    # 获取所有 GPU 的内存使用情况
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # 将 GPU 内存使用情况转为数组
    gpu_memory_array=($gpu_memory)

    # 检查是否所有 GPU 的内存使用量都小于 200MB
    all_gpus_low_memory=true
    for usage in "${gpu_memory_array[@]}"; do
        if [ "$usage" -ge 200 ]; then
            echo "All GPUs are working."
            all_gpus_low_memory=false
            break
        fi
    done

    # 如果所有 GPU 内存使用量都小于 200MB，执行 Python 文件
    if $all_gpus_low_memory; then
        echo "All GPUs have low memory. Executing Python file."
        # Put script here
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        CUDA_VISIBLE_DEVICES=0,1 \
        python eval.py \
        --model_name '12_23_mask10_cnn2' \
        --batch_size 400 \
        --num_workers 8 \
        --threshold 0.3 
        # 用你的 Python 文件替换 "your_script.py"
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        break  # 可选择退出循环
    fi

    # 等待 30 秒
    sleep 30
done
