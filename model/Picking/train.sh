CUDA_VISIBLE_DEVICES=0 \
python train.py \
--model_name '12_23_mask10_cnn2' \
--batch_size 64 \
--num_workers 4 \
--epochs 10 \
--test_mode 'true' \
