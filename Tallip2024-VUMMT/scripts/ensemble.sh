#!/bin/bash

# Decodes all test sets for a given pretrained checkpoint
# Check the Checkpoint's folder to see the created folders that contain the
# hypotheses and refs.
# CKPT="$1"
SRC="de"
TGT="en"
# if [ -z $CKPT ]; then
#   echo 'You need to provide a checkpoint .pth file for decoding.'
#   exit 1
# fi

# shift 1


ROOT="/data1/home/turghun/project/VMLM" 
DATA_PATH="${ROOT}/data/multi30k-${TGT}-${SRC}-hole"
FEAT_PATH="/data1/home/turghun/project/images/features/faster/multi30k_oidv4_features/features"
CHECK="${ROOT}/models/${TGT}-${SRC}/confusion/ummt-fintune-vmlm-en-de/vtxqruiaw0"
BS=${BS:-8}
NAME="${CKPT/.pth/}_beam${BS}/"
DUMP_PATH="${ROOT}/translation"
TEST=test
BCS=16

echo "===================== Model path:${CHECK} ======================= "

export CUDA_VISIBLE_DEVICES=3

  echo ""
  echo "=======================$TEST======================"
  echo ""

    cat $DATA_PATH/test.de-en.en | \
    python ../translate_ensemble.py --exp_name translate_ensemble \
    --exp_id en-de \
    --src_lang en --tgt_lang de \
    --model_path $CHECK --num_checkpoints 1 --output_path $DUMP_PATH \
    --beam 8 --length_penalty 1 --batch_size ${BCS} \
    --feat_path $FEAT_PATH --img_name $DATA_PATH --split test \
    --region_num 36 