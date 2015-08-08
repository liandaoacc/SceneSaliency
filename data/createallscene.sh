#!/usr/bin/env sh

SceneSaliencyBuildROOT=/home/humt/SoftwareProgram/SceneSaliency/Builds/SceneSaliency
SceneSaliencySourceROOT=/home/humt/SoftwareProgram/SceneSaliency/Sources/SceneSaliency
iSUNROOT=/home/humt/Devices/sdb5/iSUN
DATA=$iSUNROOT/data
IMAGE=$DATA/image/
SALIENCY=$DATA/saliency/

echo "Creating train lmdb..."

GLOG_logtostderr=1 $SceneSaliencyBuildROOT/tools/convert_isun_data \
    --shuffle \
    --backend=lmdb \
    --resize_width=96 \
    --resize_height=96 \
    $IMAGE \
    test_X \
    $SceneSaliencySourceROOT/data/test_input_lmdb

GLOG_logtostderr=1 $SceneSaliencyBuildROOT/tools/convert_isun_matlab_data \
    --shuffle \
    --backend=lmdb \
    --resize_width=48 \
    --resize_height=48 \
    $SALIENCY \
    test_Y \
    $SceneSaliencySourceROOT/data/test_truth_lmdb
    
echo "Done."