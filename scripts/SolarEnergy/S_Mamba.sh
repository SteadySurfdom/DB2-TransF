export CUDA_VISIBLE_DEVICES=0

model_name=DB2-TransF

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/Solar/' \
    --data_path 'solar_AL.txt' \
    --model_id 'solar_96_96' \
    --model $model_name \
    --data 'Solar' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 512 \
    --d_state 32 \
    --train_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.00005 \
    --itr 1 \
    --levels 1 \
    --use_amp

# python -u run.py \
#     --is_training 1 \
#     --root_path '../S-Mamba_datasets/Solar/' \
#     --data_path 'solar_AL.txt' \
#     --model_id 'solar_96_192' \
#     --model $model_name \
#     --data 'Solar' \
#     --features 'M' \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 2 \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --d_model 768 \
#     --d_ff 512 \
#     --d_state 32 \
#     --train_epochs 100 \
#     --batch_size 32 \
#     --learning_rate 0.00005 \
#     --itr 1 \
#     --levels 1 \
#     --use_amp

# python -u run.py \
#     --is_training 1 \
#     --root_path '../S-Mamba_datasets/Solar/' \
#     --data_path 'solar_AL.txt' \
#     --model_id 'solar_96_336' \
#     --model $model_name \
#     --data 'Solar' \
#     --features 'M' \
#     --seq_len 96 \
#     --pred_len 336 \
#     --e_layers 2 \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --d_model 768 \
#     --d_ff 512 \
#     --d_state 32 \
#     --train_epochs 100 \
#     --batch_size 32 \
#     --learning_rate 0.00005 \
#     --itr 1 \
#     --levels 1 \
#     --use_amp

# python -u run.py \
#     --is_training 1 \
#     --root_path '../S-Mamba_datasets/Solar/' \
#     --data_path 'solar_AL.txt' \
#     --model_id 'solar_96_720' \
#     --model $model_name \
#     --data 'Solar' \
#     --features 'M' \
#     --seq_len 96 \
#     --pred_len 720 \
#     --e_layers 2 \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --d_model 768 \
#     --d_ff 512 \
#     --d_state 32 \
#     --train_epochs 100 \
#     --batch_size 32 \
#     --learning_rate 0.00005 \
#     --itr 1 \
#     --levels 1 \
#     --use_amp
