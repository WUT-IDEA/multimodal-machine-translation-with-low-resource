#!/bin/bash

SRC=en
TGT=de # de, fr

SAVE_DIR=results/$SRC-$TGT/mmt

python3 average_checkpoints.py \
			--inputs ${SAVE_DIR} \
			--num-epoch-checkpoints 2 \
			--output ${SAVE_DIR}/model.pt \



