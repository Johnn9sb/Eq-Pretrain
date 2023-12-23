# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python train.py \
# --model_name '12_23_mask10_cnn2' \
# --batch_size 256 \
# --num_workers 16 \
# --epochs 10 \
# --checkpoint_path '../../pretrain/new_mask10/checkpoint.pt' 

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python train.py \
# --model_name '12_23_mask7_cnn2' \
# --batch_size 256 \
# --num_workers 16 \
# --epochs 10 \
# --checkpoint_path '../../pretrain/new_mask7/checkpoint.pt' 

CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
--model_name '12_23_mask10_cnn4' \
--batch_size 256 \
--num_workers 8 \
--epochs 10 \
--checkpoint_path '../../pretrain/new_cnn4/checkpoint.pt' 