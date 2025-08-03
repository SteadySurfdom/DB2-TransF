export CUDA_VISIBLE_DEVICES=0

model_name=DB2-TransF

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/electricity/' \
    --data_path 'electricity.csv' \
    --model_id 'ECL_96_96' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 \
    --levels 6 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/electricity/' \
    --data_path 'electricity.csv' \
    --model_id 'ECL_96_192' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 4 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 \
    --levels 1 \
    --use_amp
  
python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/electricity/' \
    --data_path 'electricity.csv' \
    --model_id 'ECL_96_336' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 4 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 \
    --levels 1 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/electricity/' \
    --data_path 'electricity.csv' \
    --model_id 'ECL_96_720' \
    --model $model_name \
    --data 'custom' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 4 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 \
    --levels 4 \
    --use_amp