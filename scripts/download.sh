#! /bin/bash
# Copyright (c) 2016 Shunta Saito

if [ ! -d data ]; then
    mkdir data
fi
cd data

# get FLIC-full dataset and FLIC-Plus annotations
if [ ! -f FLIC-full.zip ]; then
    wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
fi

if [ ! -f tr_plus_indices.mat ]; then
    wget 
fi

# Get LSP Extended Training Dataset
if [ ! -d lspet_dataset.zip ]; then
    wget http://sam.johnson.io/research/lspet_dataset.zip
fi

