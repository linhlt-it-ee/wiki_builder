export CUDA_VISIBLE_DEVICES=2

python3 main.py \
    --data_dir ../data/google_patents/us-25000 \
    --exp_name "rgat_remove-elaborate-edge" \
    --model_name rgat \
    --par1_num 300 \
    --par2_num 100 \
    --par3_num 20 \
    --epochs 20000 \
    --n_layers 3 \
    --threshold 0.2 \
    --lr 0.1
