export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --data_dir ../data/google_patents/us-25000 \
    --exp_name "test" \
    --feature_type word,cluster \
    --seed 1 \
    --model_name rgcn \
    --par1_num -1 \
    --par2_num -1 \
    --par3_num -1 \
    --hidden_feat 256 \
    --epochs 1000 \
    --lr 0.1
