#!/bin/bash

SRC=en
TGT=fr # de, fr

SAVE_DIR=results/$SRC-$TGT/mmt

python3 average_checkpoints.py \
			--inputs ${SAVE_DIR} \
			--num-epoch-checkpoints 7\
			--output ${SAVE_DIR}/model.pt \



