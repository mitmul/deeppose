#! /bin/bash
caffe_dir=$HOME/Libraries/caffe

if [ ! -d results ]; then
    mkdir results
fi
if [ ! -f data/image_mean.binaryproto ]; then
    $caffe_dir/build/tools/compute_image_mean data/image_train.lmdb data/image_mean.binaryproto
fi

cd results
fn=`date +"%Y-%m-%d_%I-%M-%S"`
mkdir $1_$fn

cd $1_$fn
cp ../../models/$1/* ./
mkdir snapshots
mkdir weights
$caffe_dir/python/draw_net.py train_test.prototxt net.png
nohup $caffe_dir/build/tools/caffe train \
    -solver=../../models/$1/solver.prototxt &
