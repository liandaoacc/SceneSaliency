#!/usr/bin/env sh
iSUNROOT=/home/humt/Devices/sdb5/iSUN
datafolder=$iSUNROOT/data
imagefolder=$datafolder/image
saliencyfolder=$datafolder/saliency
mytraining=mytraining
myvalidation=myvalidation

resize_imageW=96
resize_imageH=96
resize_saliencyW=48
resize_saliencyH=48

python createdataset.py --shuffle --check_size  --resize_imageW $resize_imageW --resize_imageH $resize_imageH --resize_saliencyW $resize_saliencyW --resize_saliencyH $resize_saliencyH --imagefolder $imagefolder --saliencyfolder $saliencyfolder --listfile $mytraining mytraining.hdf5

python createdataset.py --shuffle --check_size --resize_imageW $resize_imageW --resize_imageH $resize_imageH --resize_saliencyW $resize_saliencyW --resize_saliencyH $resize_saliencyH --imagefolder $imagefolder --saliencyfolder $saliencyfolder --listfile $myvalidation myvalidation.hdf5
