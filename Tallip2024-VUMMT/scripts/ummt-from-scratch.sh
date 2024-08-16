#!/bin/bash

# Trains an UMMT from scratch on Multi30k


tgt="$1"


SRC=zh
TGT=${tgt}

ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k-${SRC}-${TGT}-half"
FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/orign"

DUMP_PATH=/data1/home/turghun/project/low/models/${SRC}-${TGT}



EPOCH_SIZE=14500

export CUDA_VISIBLE_DEVICES=2

python ../train.py  --beam_size 8 --exp_name mbt-ummt-scratch --dump_path ${DUMP_PATH} \
    --data_path "${DATA_PATH}"  --reload_model "" \
    --lgs "${SRC}-${TGT}"  --vbt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}"  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
    --batch_size 64  --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8  \
    --dropout 0.1 --attention_dropout 0.1  --gelu_activation true \
    --inputs_concat true --select_attn false  \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 --keep_best_checkpoints 11 \
    --epoch_size ${EPOCH_SIZE}  --eval_bleu true  --stopping_criterion "valid_${SRC}-${TGT}_mmt_bleu,10" \
    --validation_metrics "valid_${SRC}-${TGT}_mmt_bleu,valid_${TGT}-${SRC}_mmt_bleu" \
    --image_names $DATA_PATH  --region_feats_path $FEAT_PATH --num_of_regions 36 



