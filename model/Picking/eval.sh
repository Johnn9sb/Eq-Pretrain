CUDA_VISIBLE_DEVICES=0,1 \
python eval.py \
--model_name '12_23_mask10_cnn2' \
--batch_size 400 \
--num_workers 8 \
--threshold 0.2 

# CUDA_VISIBLE_DEVICES=0,1 \
# python eval.py \
# --model_name '12_23_mask7_cnn2' \
# --batch_size 400 \
# --num_workers 8 \
# --threshold 0.2 \