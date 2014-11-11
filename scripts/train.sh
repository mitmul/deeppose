#! /bin/bash
if [ ! -d results ]; then
    mkdir results
fi
cd results
fn=`date +"%Y-%m-%d_%I-%M-%S"`
mkdir $1_$fn
cd $1_$fn
cp ../../models/$1/* ./

caffe_dir=$HOME/Libraries/caffe
$caffe_dir/python/draw_net.py train_test.prototxt net.png
nohup $caffe_dir/build/tools/caffe train \
    -solver=../../models/$1/solver.prototxt &
