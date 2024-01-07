CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python train.py \
--model_name '1_5_768_3000_linear' \
--batch_size 64 \
--num_workers 4 \
--epochs 10 \
--decoder_type 'linear' \
--checkpoint_path '/mnt/nas3/johnn9/Eq-Pretrain/pretrain/mask7_d768/checkpoints/checkpoint_3000.pt' \
# --test_mode 'true'
# --weight 3.0 \
