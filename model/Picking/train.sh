# CUDA_VISIBLE_DEVICES=0,1 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name '2_1_768_scratch_linear' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --decoder_type 'linear' \
# --checkpoint_path 'None' \

# --test_mode 'true' \
# --weight 3.0 \

CUDA_VISIBLE_DEVICES=2,3 \
exec -a Johnn9_Finetune \
python train.py \
--model_name 'eqt' \
--batch_size 64 \
--num_workers 4 \
--epochs 200 \
--train_model 'eqt' \
# --test_mode 'true' \

# CUDA_VISIBLE_DEVICES=2,3 \
# exec -a Johnn9_Finetune \
# python train.py \
# --model_name 'phasenet' \
# --batch_size 64 \
# --num_workers 4 \
# --epochs 200 \
# --train_model 'phasenet' \