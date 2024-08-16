#!/bin/bash

# Train unmt form scratch


tgt="$1"

SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project/low"
DATA_PATH="/data1/home/turghun/project/VMLM/data/multi30k-${SRC}-${TGT}-half-uy"

DUMP_PATH=${ROOT}/models/${SRC}-${TGT}/

EPOCH_SIZE=14500
export CUDA_VISIBLE_DEVICES=0
BSZ=64



python ../train.py  --beam_size 8 --exp_name unmt-scratch  --dump_path ${DUMP_PATH} \
    --data_path "${DATA_PATH}" --reload_model "" \
    --lgs "${SRC}-${TGT}"  --ae_steps "${SRC},${TGT}"  --bt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}" --word_shuffle 3  \
    --word_dropout 0.1  --word_blank 0.1  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
    --tokens_per_batch 2000 --batch_size $BSZ  --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8  \
    --dropout 0.1 --attention_dropout 0.1  --gelu_activation true \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 \
    --epoch_size ${EPOCH_SIZE}  --eval_bleu true  --stopping_criterion "valid_${SRC}-${TGT}_mt_bleu,10" \
    --validation_metrics "valid_${SRC}-${TGT}_mt_bleu,valid_${TGT}-${SRC}_mt_bleu" 