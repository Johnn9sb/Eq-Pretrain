# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python threshold.py \
# --model_name '2_7_scratch_cnn_detect' \
# --batch_size 32 \
# --num_workers 4 \
# --parl 'y' \
# --task 'detect' \
# --decoder_type 'cnn' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=0 \
exec -a Johnn9_Finetune \
python threshold.py \
--model_name '2_22_lem35000_frez_cnn' \
--train_model 'wav2vec2' \
--batch_size 32 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
# --test_mode 'true' \


# CUDA_VISIBLE_DEVICES=0 \
# python threshold.py \
# --model_name '12_23_mask10_cnn4' \
# --batch_size 400 \
# --num_workers 8 \