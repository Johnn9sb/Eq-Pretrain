# CUDA_VISIBLE_DEVICES=0,1 \
# python threshold.py \
# --model_name '12_23_mask10_cnn2' \
# --batch_size 400 \
# --num_workers 8 \

CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python threshold.py \
--model_name '2_5_scratch_cnn' \
--train_model 'wav2vec2' \
--batch_size 32 \
--num_workers 4 \
--parl 'y' \
--decoder_type 'cnn' \

# --test_mode 'true' \


# CUDA_VISIBLE_DEVICES=0 \
# python threshold.py \
# --model_name '12_23_mask10_cnn4' \
# --batch_size 400 \
# --num_workers 8 \