#!/usr/bin/env bash

for embed_dim in 100;
do
    mkdir -p "./ckpts_${embed_dim}/"

    /netscratch/samin/dev/miniconda3/envs/dsre-vae/bin/dglke_train --model_name TransE_l2 \
                --dataset UMLS \
                --data_path . \
                --format raw_udd_hrt \
                --data_files case4_train.txt case4_valid.txt case4_test.txt \
                --batch_size 1024 \
                --neg_sample_size 256 \
                --hidden_dim ${embed_dim} \
                --gamma 10 \
                --lr 0.1 \
                --regularization_coef 1e-7 \
                --batch_size_eval 1000 \
                --test -adv \
                --gpu 0 \
                --max_step 100000 \
                --neg_sample_size_eval 1000 \
                --log_interval 1000 \
                --save_path "./ckpts_${embed_dim}/"
done
