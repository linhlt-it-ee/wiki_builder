export CUDA_VISIBLE_DEVICES=1
python3 main.py \
    --pretrained_model_name distilbert-base-uncased \
    --data_dir ../data/google_patents/us-25000 \
    --hidden_dim 128 \
    --num_train_epochs 200 \
    --lr 1e-2 \
    --log_path "runs/test"
