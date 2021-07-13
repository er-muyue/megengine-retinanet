#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/
export MEGENGINE_LOGGING_LEVEL=ERROR

# ---- Train ---- #
# python3 mge_tools/train.py -n 2 -b 2 -f configs/retinanet_res50_3x_800size_chongqigongmen.py -d data

# ---- Test ---- #
# python3 mge_tools/test.py -n 1 -se 35 -f configs/retinanet_res50_3x_800size_chongqigongmen.py -d data

# ----Train Search ---- #
# python3 mge_tools/train_search.py -cfg configs/retinanet_res50_3x_800size_chongqigongmen.py -hpo configs/search_config.yaml

rlaunch --cpu=4 --gpu=2 --memory=40960 -- python3 tools/train_search.py -cfg configs/retinanet_res50_3x_800size_chongqigongmen.py -hpo configs/search_config.yaml
# inference
#python3 tools/inference.py \
#   -f configs/retinanet_res50_3x_800size_chongqigongmen.py  \
#   -i data/chongqigongmen/images/18516456,19ce1000ac177dec.jpg \
#   -w log-of-retinanet_res50_3x_800size_chongqigongmen/epoch_35.pkl
