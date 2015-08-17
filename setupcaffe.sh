#!/usr/bin/env sh
# $1 absolute path to caffe, eg. /path/caffe
caffe_root=$1
caffe_folder=$(basename $caffe_root)

scenesaliency_root=$(cd "$(dirname "$0")"; pwd -P)

if [ ! -d $caffe_root ]; then
  echo $caffe_root "is not exist"
  exit
fi

if [ ! -L $scenesaliency_root/caffe ]; then
  echo "Create symbolic links relative to " $caffe_root
  ln -s $caffe_root $scenesaliency_root/caffe
fi
