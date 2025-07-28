export CUDA_VISIBLE_DEVICES=0

model_name=DB2-TransF

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/exchange_rate/' \
    --data_path 'exchange_rate.csv' \
    --model_id 'Exchange_96_96' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 96 \
    --d_ff 128 \
    --d_state 32 \
    --train_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --itr 1 \
    --levels 4 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/exchange_rate/' \
    --data_path 'exchange_rate.csv' \
    --model_id 'Exchange_96_192' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 96 \
    --d_ff 128 \
    --d_state 32 \
    --train_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --itr 1 \
    --levels 4 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/exchange_rate/' \
    --data_path 'exchange_rate.csv' \
    --model_id 'Exchange_96_336' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 96 \
    --d_ff 128 \
    --d_state 32 \
    --train_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --itr 1 \
    --levels 4 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/exchange_rate/' \
    --data_path 'exchange_rate.csv' \
    --model_id 'Exchange_96_720' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 96 \
    --d_ff 128 \
    --d_state 32 \
    --train_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --itr 1 \
    --levels 1 \
    --use_amp