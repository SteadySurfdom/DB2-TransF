export CUDA_VISIBLE_DEVICES=0

model_name=DB2-TransF

python -u run.py \
  --is_training 1 \
  --root_path '../S-Mamba_datasets/PEMS/' \
  --data_path 'PEMS04.npz' \
  --model_id 'PEMS04_96_12' \
  --model $model_name \
  --data 'PEMS' \
  --features 'M' \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 8 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 1536 \
  --d_ff 1536 \
  --d_state 32 \
  --train_epochs 100 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --itr 1 \
  --levels 1 \
  --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/PEMS/' \
    --data_path 'PEMS04.npz' \
    --model_id 'PEMS04_96_24' \
    --model $model_name \
    --data 'PEMS' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 8 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1536 \
    --d_ff 1536 \
    --d_state 32 \
    --train_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.003 \
    --itr 1 \
    --levels 1 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/PEMS/' \
    --data_path 'PEMS04.npz' \
    --model_id 'PEMS04_96_48' \
    --model $model_name \
    --data 'PEMS' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 8 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1536 \
    --d_ff 1536 \
    --d_state 32 \
    --train_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0035 \
    --itr 1 \
    --levels 1 \
    --use_amp

python -u run.py \
    --is_training 1 \
    --root_path '../S-Mamba_datasets/PEMS/' \
    --data_path 'PEMS04.npz' \
    --model_id 'PEMS04_96_96' \
    --model $model_name \
    --data 'PEMS' \
    --features 'M' \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 8 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1536 \
    --d_ff 1536 \
    --d_state 32 \
    --train_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.004 \
    --itr 1 \
    --levels 1 \
    --use_amp
