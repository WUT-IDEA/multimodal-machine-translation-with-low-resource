#!/bin/bash

# Takes a TLM checkpoint and finetunes it for Multi30k NMT (no visual features)


PRE="$1"

if [ -z $PRE ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

shift 1

SRC=zh
TGT=uy

ROOT="/data1/home/turghun/project/VMLM"
TLM_PATH="${ROOT}/pretrained/zh-en/best-valid_${PRE}_ppl.pth"
DATA_PATH="${ROOT}/data/multi30k-zh-uy-half"

EPOCH=14500
DUMP_PATH=${ROOT}/models/${SRC}-${TGT}/
NAME="nmt-fintune-${PRE}"

export CUDA_VISIBLE_DEVICES=2

echo "-----------------model is initiaolized by ${TLM_PATH}---------------------"

python ../train.py --beam_size 8 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --reload_model "${TLM_PATH},${TLM_PATH}" --data_path ${DATA_PATH} --encoder_only false \
  --lgs "${SRC}-${TGT}" --mt_step "${SRC}-${TGT}" --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001" \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 \
  --stopping_criterion "valid_${SRC}-${TGT}_mt_bleu,20" --validation_metrics "valid_${SRC}-${TGT}_mt_bleu" $@
