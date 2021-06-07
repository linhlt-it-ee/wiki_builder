export CUDA_VISIBLE_DEVICES=2
python3 main.py \
    --pretrained_model_name distilbert-base-uncased \
    --data_dir ../data/reuters \
    --hidden_dim 256 \
    --num_train_epochs 5000 \
    --lr 1e-2 \
    --threshold 0.4 \
    --log_path "runs/epoch=5000_hidden_dim=256_lr=1e-2_threshold=0.4"
