CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py \
--model_name '12_23_mask10_cnn2' \
--batch_size 256 \
--num_workers 16 \
--epochs 10 \
# --test_mode 'true' \
