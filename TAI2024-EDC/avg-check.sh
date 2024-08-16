#!/bin/bash

SRC=en
TGT=fr # de

python3 scripts/average_checkpoints.py \
			--inputs results/$SRC-$TGT \
			--num-epoch-checkpoints 10 \
			--output results/$SRC-$TGT/model.pt \
