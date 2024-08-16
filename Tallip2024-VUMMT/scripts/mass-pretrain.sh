#!/bin/bash



tgt="$1"


SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k-${SRC}-${TGT}-half-uy"
DUMP_PATH=${ROOT}/models/mass/${SRC}-${TGT}/

EPOCH_SIZE=14500


export CUDA_VISIBLE_DEVICES=0

python ../train.py --exp_name mass-multi30k-${SRC}-${TGT} --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --mass_steps "${SRC},${TGT}" --encoder_only false \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 256 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001" \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --eval_bleu true  --validation_metrics "valid_${SRC}-${TGT}_mt_bleu" \
  --fp16 false --keep_best_checkpoints 11 --word_mass 0.5 --min_len 5 \

  