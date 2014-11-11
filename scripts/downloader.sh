#! /bin/bash
if [ ! -d data ]; then
    mkdir data
    cd data
    wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
    unzip FLIC-full.zip
    rm -rf FLIC-full.zip
    cd ..
fi
python scripts/save_crops.py