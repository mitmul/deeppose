#! /bin/bash
if [ ! -d data ]; then
    mkdir data
    cd data
    wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
    unzip FLIC-full.zip
    rm -rf FLIC-full.zip
    cd FLIC-full
    wget http://cims.nyu.edu/~tompson/data/tr_plus_indices.mat
    cd ..
    wget http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip
    unzip lspet_dataset.zip
    rm -rf lspet_dataset.zip
    mkdir lspet_dataset
    mv images lspet_dataset/
    mv joints.mat lspet_dataset/
    mv README.txt lspet_dataset/
    cd ..
fi
