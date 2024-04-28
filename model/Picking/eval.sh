# CUDA_VISIBLE_DEVICES=0,1 \
# exec -a Johnn9_Finetune \
# python eval.py \
# --model_name 'phasenet_1000' \
# --train_model 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --threshold 0.3 \
# --dataset 'stead' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name '4_22_lempick_1000_freeze_cnn' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--threshold 0.6 \
--weighted_sum 'n' \
# --dataset 'stead' \
# --test_mode 'true' \

