# CUDA_VISIBLE_DEVICES=0,1 \
# exec -a Johnn9_Finetune \
# python threshold.py \
# --model_name 'phasenet_1000' \
# --train_model 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --parl 'y' \
# --task 'pick' \
# --dataset 'stead' \
# --test_mode 'true' \

CUDA_VISIBLE_DEVICES=3 \
exec -a Johnn9_Finetune \
python threshold.py \
--model_name 'data2vecpick_wei' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--weighted_sum 'y' \
--dataset 'stead' \
# --test_mode 'true' \

