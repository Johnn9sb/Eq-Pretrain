# CUDA_VISIBLE_DEVICES=0,1 \
# exec -a Johnn9_Finetune \
# python inference.py \
# --model_name 'eqt_1000' \
# --train_model 'eqt' \
# --batch_size 1 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --threshold 0.5 \

# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=0,1 \
exec -a Johnn9_Finetune \
python inference.py \
--model_name '3_4_lem50000_wei_cnn' \
--train_model 'wav2vec2' \
--batch_size 1 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--threshold 0.4 \
--weighted_sum 'y'\

# --test_mode 'true' \

# CUDA_VISIBLE_DEVICES=2 \
# exec -a Johnn9_Finetune \
# python inference.py \
# --model_name 'eqt' \
# --train_model 'eqt' \
# --batch_size 1 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --threshold 0.5 \
