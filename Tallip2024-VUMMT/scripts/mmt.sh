#!/bin/bash

# Trains an NMT from scratch on Multi30k

SRC=en
TGT=de

ROOT="/data1/home/turghun/project"

FEAT_PATH="$ROOT/images/coco2014-multi30k/features/faster_oidv4_features"
DATA_PATH="$ROOT/VMLM/data/mscoco/multi30k/mono/${SRC}-${TGT}"
DUMP_PATH=$ROOT/acmmm/models/${SRC}-${TGT}

EXP_NAME="mmt-from-scratch-${SRC}-${TGT}-concat-exam"
EPOCH_SIZE=29000

export CUDA_VISIBLE_DEVICES=2

python ../train.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --encoder_only false \
  --lgs "${SRC}-${TGT}" --mmt_step "${SRC}-${TGT}" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --inputs_concat true --select_attn false  \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --keep_best_checkpoints 11 \
  --stopping_criterion "valid_${SRC}-${TGT}_mmt_bleu,20" --validation_metrics "valid_${SRC}-${TGT}_mmt_bleu" \
  --iter_seed 12345 --region_feats_path $FEAT_PATH --image_names $DATA_PATH \
  --visual_first true --num_of_regions 36 $@