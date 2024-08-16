#!/bin/bash

SRC=en
TGT=fr # de, fr

SAVE_DIR=results/nmt/$SRC-$TGT

python3 average_checkpoints.py \
			--inputs ${SAVE_DIR} \
			--num-epoch-checkpoints 3 \
			--output ${SAVE_DIR}/model.pt \



