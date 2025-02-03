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

CUDA_VISIBLE_DEVICES=0,1 \
exec -a Johnn9_Finetune \
python eval.py \
--model_name 'data2vecpick_wei_1000' \
--train_model 'wav2vec2' \
--batch_size 64 \
--num_workers 4 \
--decoder_type 'cnn' \
--parl 'y' \
--task 'pick' \
--threshold 0.4 \
--weighted_sum 'y' \
--dataset 'hualien' \
# --noise_need 'false' \
# --test_mode 'true' \

