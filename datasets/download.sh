#! /bin/bash
# Copyright (c) 2016 Shunta Saito

if [ ! -d data ]; then
    mkdir data
fi
cd data

# get FLIC-full dataset and FLIC-Plus annotations
if [ ! -f FLIC-full/tr_plus_indices.mat ]; then
    wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
    unzip FLIC-full.zip
    rm -rf FLIC-full.zip
    cd FLIC-full
    wget https://cims.nyu.edu/~tompson/data/tr_plus_indices.mat
    cd ..
fi

# Get LSP Extended Training Dataset
if [ ! -d lspet_dataset ]; then
    wget http://sam.johnson.io/research/lspet_dataset.zip
    unzip lspet_dataset.zip
    rm -rf lspet_dataset.zip
    mkdir lspet_dataset
    mv images lspet_dataset/
    mv joints.mat lspet_dataset/
    mv README.txt lspet_dataset/
fi

