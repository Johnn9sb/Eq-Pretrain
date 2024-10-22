CUDA_VISIBLE_DEVICES=2,3 \
python test.py \
--save_path '/mnt/nas3/johnn9/mag_checkpoint/lem_scratch' \
--model_opt 'w2v' \
--decoder_type 'cnn' \
--batch_size 64 \
--workers 4 \
--level 4 \
--without_noise 'False' \
# --dataset_opt 'stead' \

# CUDA_VISIBLE_DEVICES=0 \
# python test.py \
# --save_path '/mnt/nas3/johnn9/mag_checkpoint/magnet_1000' \
# --model_opt 'magnet' \
# --batch_size 64 \
# --workers 4 \
# --level 4 \
# --without_noise 'False' \