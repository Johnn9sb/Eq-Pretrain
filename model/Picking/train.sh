# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name 'eqt_1000' \
# --train_model 'eqt' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --lr 0.0001 \
# --task 'pick' \
# --resume 'true' \

CUDA_VISIBLE_DEVICES=0,1 \
exec -a Johnn9_Finetune \
python train.py \
--model_name '4_28_lempick_1000_wei_cnn' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--epochs 200 \
--decoder_type 'cnn' \
--freeze 'y' \
--weighted_sum 'y' \
--lr 0.00005 \
--task 'pick' \
--checkpoint_path "/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt" \

# --resume 'true' \
# --checkpoint_path "None" \
