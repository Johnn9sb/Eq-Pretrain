# CUDA_VISIBLE_DEVICES=0,1 \
# python threshold.py \
# --model_name '12_23_mask10_cnn2' \
# --batch_size 400 \
# --num_workers 8 \

CUDA_VISIBLE_DEVICES=0,1 \
exec -a Johnn9_Finetune \
python threshold.py \
--model_name '1_4_768_15000_linear' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'linear' \

# CUDA_VISIBLE_DEVICES=0 \
# python threshold.py \
# --model_name '12_23_mask10_cnn4' \
# --batch_size 400 \
# --num_workers 8 \