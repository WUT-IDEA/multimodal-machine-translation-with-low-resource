#!/bin/bash

# Trains TLM on Multi30k for quick / demonstrational purposes

SRC=uy
TGT=zh

ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k-${TGT}-${SRC}-hole"

DUMP_PATH=${ROOT}/models/supervised/${SRC}-${TGT}/
EXP_NAME="tlm-multi30k"
EPOCH_SIZE=29000

export CUDA_VISIBLE_DEVICES=3

python ../train.py --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --data_path ${DATA_PATH}\
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC}-${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 1 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics "_valid_${SRC}_${TGT}_mlm_ppl" --stopping_criterion "_valid_${SRC}_${TGT}_mlm_ppl,50" \
  --fp16 false --save_periodic 5 --iter_seed 12345 $@
