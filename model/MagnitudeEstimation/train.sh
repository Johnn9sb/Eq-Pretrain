CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
--save_path '/mnt/nas3/johnn9/mag_checkpoint/lem_freeze_50000' \
--model_opt 'w2v' \
--decoder_type 'cnn' \
--batch_size 64 \
--epochs 200 \
--lr 0.0001 \
--workers 4 \
--level 4 \
--w2v_path '/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt' \
--without_noise 'False' \

