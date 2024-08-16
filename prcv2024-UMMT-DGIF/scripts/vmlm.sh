#!/bin/bash



tgt="$1"


SRC=en
TGT=${tgt}

ROOT="/data1/home/turghun/project"

REGION_FEAT_PATH="$ROOT/images/coco2014-multi30k/features/faster_oidv4_features"
GRID_FEAT_PATH="$ROOT/images/features/resnet101-local"
GLOBAL_FEAT_PATH="$ROOT/images/features/resnet50-global"


DATA_PATH="$ROOT/VMLM/data/multi30k/mono/${SRC}-${TGT}"
DUMP_PATH=${ROOT}/low/models/${SRC}-${TGT}
EXP_NAME="vmlm-multi30k-region"

REGION=true
GRID=false
GLOBAL=false

EPOCH_SIZE=14500
BSZ=64

export CUDA_VISIBLE_DEVICES=2

python ../train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size $BSZ --bptt 256 \
  --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 --fp16 false --keep_best_checkpoints 11 \
  --validation_metrics '_valid_vmlm_ppl' --stopping_criterion '_valid_vmlm_ppl,50' \
  --image_names $DATA_PATH  --region_feats_path $REGION_FEAT_PATH --grid_feats_path $GRID_FEAT_PATH\
  --global_feats_path $GLOBAL_FEAT_PATH --inputs_concat true  --num_of_regions 36 \
  --re_img_feats $REGION --gr_img_feats $GRID --gl_img_feats $GLOBAL \
  --only_vmlm true  --eval_vmlm true --region_mask_type mask --image_text_con true 