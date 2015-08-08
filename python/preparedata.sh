#!/usr/bin/env sh
iSUNROOT=/home/humt/Devices/sdb5/iSUN
datafolder=$iSUNROOT/data
imagefolder=$datafolder/image
saliencyfolder=$datafolder/saliency
listfile=test

python convertdata.py --imagefolder $imagefolder \
--saliencyfolder $saliencyfolder --listfile $listfile \
testoutput
