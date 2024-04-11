CUDA_VISIBLE_DEVICES=1 \
python test.py \
--save_path '/mnt/nas3/johnn9/mag_checkpoint/lem_freeze_50000' \
--model_opt 'w2v' \
--decoder_type 'cnn' \
--batch_size 64 \
--workers 4 \
--level 4 \
--without_noise 'False' \
--w2v_path '/mnt/nas3/johnn9/pretrain/11-54-34/checkpoints/checkpoint_46_50000.pt' \ 