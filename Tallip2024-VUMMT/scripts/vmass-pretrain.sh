#!/bin/bash

# Trains VTLM on Multi30k for quick / demonstrational purposes


SRC=en
TGT=uy

ROOT="/data1/home/turghun/project/VMLM"
DATA_PATH="${ROOT}/data/multi30k-${SRC}-${TGT}-half-la"
FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/features"

DUMP_PATH=${ROOT}/models/vmass/${SRC}-${TGT}/latin
EXP_NAME="vmass-multi30k-s"
EPOCH_SIZE=14500

export CUDA_VISIBLE_DEVICES=3

# --region_mask_type mask: Replaces masked region feature vectors with [MASK] embedding
# --region_mask_type zero: Replaces masked region feature vectors with a 0-vector


python ../train.py --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --data_path ${DATA_PATH}\
  --lgs "${SRC}-${TGT}" --vmass_steps "${SRC},${TGT}" --beam_size 8 \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' --encoder_only false \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 256 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001" \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 --tokens_per_batch 2000 \
  --eval_bleu true  --validation_metrics "valid_${SRC}-${TGT}_mmt_bleu" --inputs_concat false \
  --fp16 false --iter_seed 12345 --word_mass 0.5 --min_len 5 \
  --image_names $DATA_PATH --region_feats_path $FEAT_PATH --visual_first true --select_attn true \
  --num_of_regions 36  --region_mask_type mask $@
