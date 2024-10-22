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

CUDA_VISIBLE_DEVICES=0 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name '3_4_lem50000_wei_cnn' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--threshold 0.5 \
--weighted_sum 'y' \
--noise_need 'false' \
# --dataset 'stead' \
# --test_mode 'true' \

