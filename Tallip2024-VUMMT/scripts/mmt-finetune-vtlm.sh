#!/bin/bash

# Takes a VTLM checkpoint and finetunes it for Multi30k MMT


PRE="$1"

if [ -z $PRE ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

shift 1

SRC=en
TGT=uy

ROOT="/data1/home/turghun/project/VMLM"
VTLM_PATH="${ROOT}/pretrained/supervised/uy-en/best-valid_${PRE}_ppl.pth"
DATA_PATH="${ROOT}/data/multi30k-en-uy-hole-uy"
FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/features"

DUMP_PATH=${ROOT}/models/supervised/${SRC}-${TGT}/
NAME="mmt-fintune-${PRE}_${SRC}-${TGT}-concat"

EPOCH=29000

echo "-----------------model is initiaolized by ${VTLM_PATH}---------------------"

export CUDA_VISIBLE_DEVICES=0

python ../train.py --beam_size 8 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --reload_model "${VTLM_PATH},${VTLM_PATH}" --data_path ${DATA_PATH} --encoder_only false \
  --lgs "${SRC}-${TGT}" --mmt_step "${SRC}-${TGT}" --inputs_concat true --select_attn false \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8 \
  --stopping_criterion "valid_${SRC}-${TGT}_mmt_bleu,20" --validation_metrics valid_${SRC}-${TGT}_mmt_bleu \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true \
  --num_of_regions 36 --reg_enc_bias true  $@
