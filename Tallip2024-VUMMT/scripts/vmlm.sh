#!/bin/bash


tgt="$1"


SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project"

FEAT_PATH="$ROOT/images/coco2014-multi30k/features/faster_oidv4_features"
DATA_PATH="$ROOT/VMLM/data/mscoco/mscoco-multi30k/mono/${SRC}-${TGT}-uy"
DUMP_PATH=$ROOT/acmmm/models/${SRC}-${TGT}-uy
EXP_NAME="vmlm-mscoco-multi30k-region-concat-selective"

EPOCH_SIZE=100000
BSZ=64

export CUDA_VISIBLE_DEVICES=1

python ../train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size $BSZ --bptt 256 \
  --optimizer 'adam,lr=0.0001'  \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 --fp16 false --keep_best_checkpoints 1 \
  --validation_metrics '_valid_vmlm_ppl' --stopping_criterion '_valid_vmlm_ppl,50' \
  --tokens_per_batch 2000 --image_names $DATA_PATH  --region_feats_path $FEAT_PATH \
  --inputs_concat true --select_attn true \
  --num_of_regions 36 --only_vmlm true  --eval_vmlm true --region_mask_type mask --eval_from 15