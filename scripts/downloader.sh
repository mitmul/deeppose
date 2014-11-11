#! /bin/bash
if [ ! -d data ]; then
    mkdir data
    cd data
    wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
    unzip FLIC-full.zip
    rm -rf FLIC-full.zip
    wget http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip
    unzip lspet_dataset.zip
    rm -rf lspet_dataset.zip
    mkdir lspet_dataset
    mv images lspet_dataset/
    mv joints.mat lspet_dataset/
    cd ..
fi
python scripts/save_crops.py