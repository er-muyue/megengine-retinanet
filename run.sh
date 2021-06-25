#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/

# ---- Train ---- #
python3 tools/train.py \
    -n 2 -b 2 \
    -f configs/retinanet_res50_3x_800size_chongqigongmen.py \
    -d data

# ---- Test ---- #
python3 tools/test.py \
    -n 1 -se 35 \
    -f configs/retinanet_res50_3x_800size_chongqigongmen.py  \
    -d data

# inference
#python3 tools/inference.py \
#   -f configs/retinanet_res50_3x_800size_chongqigongmen.py  \
#   -i data/chongqigongmen/images/18516456,19ce1000ac177dec.jpg \
#   -w log-of-retinanet_res50_3x_800size_chongqigongmen/epoch_35.pkl
