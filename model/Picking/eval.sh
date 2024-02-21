# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python eval.py \
# --model_name '2_7_scratch_cnn_detect' \
# --batch_size 32 \
# --num_workers 4 \
# --parl 'y' \
# --task 'detect' \
# --threshold 0.4 \
# --decoder_type 'cnn' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name 'eqt_detect' \
--train_model 'eqt' \
--batch_size 32 \
--num_workers 4 \
--parl 'y' \
--task 'detect' \
--threshold 0.4 \
# --test_mode 'true' \

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