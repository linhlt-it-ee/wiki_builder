export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --data_dir ../data/google_patents/us-25000 \
    --model_name rgcn \
    --par1_num 300 \
    --par2_num 100 \
    --par3_num 20 \
    --epochs 5000 \
    --n_layers 5 \
    --lr 0.1
