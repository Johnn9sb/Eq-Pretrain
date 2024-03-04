# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name '2_22_lem30000_frez_cnn' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --decoder_type 'cnn' \
# --checkpoint_path '/mnt/nas3/johnn9/pretrain/17-56-33/checkpoints/checkpoint_28_30000.pt' \
# --freeze 'y' \
# --lr 0.00005 \

# --checkpoint_path '/mnt/nas3/johnn9/pretrain/00-06-19/checkpoints/checkpoint_32_35000.pt' \

# --test_mode 'true' \
# --weight 3.0 \

CUDA_VISIBLE_DEVICES=2 \
exec -a Johnn9_Finetune \
python train.py \
--model_name '3_4_lem50000_wei_cnn' \
--batch_size 64 \
--num_workers 4 \
--epochs 200 \
--decoder_type 'cnn' \
--checkpoint_path "/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt" \
--freeze 'y' \
--weighted_sum 'y' \
--lr 0.0001 \
# --resume 'true' \


# CUDA_VISIBLE_DEVICES=2,3 
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name 'eqt_detect' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --train_model 'eqt' \
# --task 'detect' \
# --test_mode 'true' \

# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --train_model 'phasenet' \