CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python inference.py \
--model_name 'phasenet' \
--train_model 'phasenet' \
--batch_size 32 \
--num_workers 4 \
--parl 'y' \
--task 'pick' \
--threshold 0.5 \

# --test_mode 'true' \

# CUDA_VISIBLE_DEVICES=2 \
# exec -a Johnn9_Finetune \
# python inference.py \
# --model_name '3_26_lem50000pick_tune_cnn' \
# --train_model 'wav2vec2' \
# --batch_size 32 \
# --num_workers 4 \
# --decoder_type 'cnn' \
# --parl 'y' \
# --task 'pick' \
# --threshold 0.4 \

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
