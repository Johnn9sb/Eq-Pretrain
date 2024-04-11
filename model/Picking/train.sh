# CUDA_VISIBLE_DEVICES=3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name 'eqt_detect' \
# --train_model 'eqt' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --lr 0.00005 \
# --task 'detect' \
# --resume 'true' \

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name '3_26_lem50000pick_tune_cnn' \
# --batch_size 64 \
# --num_workers 10 \
# --epochs 200 \
# --decoder_type 'cnn' \
# --checkpoint_path "/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt" \
# --freeze 'n' \
# --weighted_sum 'n' \
# --lr 0.00001 \
# --task 'pick' \
# --resume 'true' \

CUDA_VISIBLE_DEVICES=2 \
exec -a Johnn9_Finetune \
python train.py \
--model_name '4_11_lem50000detect_freeze_cnn' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--epochs 200 \
--decoder_type 'cnn' \
--checkpoint_path "/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt" \
--freeze 'y' \
--weighted_sum 'n' \
--lr 0.00005 \
--task 'detect' \
# --resume 'true' \