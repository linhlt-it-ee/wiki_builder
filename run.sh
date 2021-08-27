export CUDA_VISIBLE_DEVICES=1

python3 main.py \
    --data_dir ../data/google_patents/us-25000 \
    --exp_name "RSAGE2_mlp_textgcn-features_seed=1" \
    --feature_type textgcn \
    --seed 1 \
    --model_name rsage2 \
    --par1_num 1000 \
    --par2_num 300 \
    --par3_num 50 \
    --epochs 2000 \
    --lr 0.1
