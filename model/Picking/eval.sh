CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name '1_4_768_10000_linear' \
--batch_size 64 \
--num_workers 4 \
--threshold 0.3 \
--decoder_type 'linear' \

# wait

# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python eval.py \
# --model_name '12_29_mask5_cnn2_linear_weight_3' \
# --batch_size 64 \
# --num_workers 4 \
# --threshold 0.2 \
# --decoder_type 'linear' \

# CUDA_VISIBLE_DEVICES=0,1 \
# python eval.py \
# --model_name '12_23_mask7_cnn2' \
# --batch_size 400 \
# --num_workers 8 \
# --threshold 0.3 \

# CUDA_VISIBLE_DEVICES=0 \
# python eval.py \
# --model_name '12_23_mask10_cnn4' \
# --batch_size 400 \
# --num_workers 8 \
# --threshold 0.2 \