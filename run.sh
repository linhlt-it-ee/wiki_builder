export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --data_path ../data/google_patents/us-25000/data_200.ndjson \
    --cache_dir ../data/google_patents/us-25000/cache_200 \
    --exp_name "report-200en|rgat2-add_weight|word,cluster-100" \
    --n_clusters 100 \
    --feature_type word,cluster \
    --seed 1 \
    --model_name rgat2 \
    --num_heads 4 \
    --hidden_feat 512 \
    --epochs 1000 \
    --dropout 0.3 \
    --lr 0.1
