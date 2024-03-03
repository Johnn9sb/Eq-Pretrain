# CUDA_VISIBLE_DEVICES=2,3 \
# python train.py \
# --save_path '/mnt/nas3/johnn9/mag_checkpoint' \
# --model_opt 'w2v' \
# --decoder_type 'cnn' \
# --batch_size 64 \
# --epochs 200 \
# --lr 0.00005 \
# --workers 4 \
# --level 4 \
# --without_noise 'False' \
# --w2v_path 'None' \

CUDA_VISIBLE_DEVICES=1 \
python test.py \
--save_path '/mnt/nas3/johnn9/ma_checkpoint' \
--model_opt 'w2v' \
--decoder_type 'cnn' \
--batch_size 32 \
--workers 4 \
--level 4 \
--without_noise 'False' \
--w2v_path 'None' \