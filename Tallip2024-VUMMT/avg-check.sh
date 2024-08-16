#!/bin/bash

EXP_NAME=fine_ummt_vmlm_ende

python3 src/average_checkpoints.py \
			--inputs models/$EXP_NAME/nqty9c1hju \
			--num-epoch-checkpoints 10 \
			--output models/$EXP_NAME/nqty9c1hju/model.pth \
