#! /bin/bash
# Copyright (c) 2016 Shunta Saito

if [ ! -d data ]; then
    mkdir data
fi
cd data

# get FLIC-full dataset and FLIC-Plus annotations
wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
unzip FLIC-full.zip
rm -rf FLIC-full.zip
cd FLIC-full
wget https://cims.nyu.edu/~tompson/data/tr_plus_indices.mat
cd ..

# Get LSP Extended Training Dataset
wget http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip
unzip lspet_dataset.zip
rm -rf lspet_dataset.zip
mkdir lspet_dataset
mv images lspet_dataset/
mv joints.mat lspet_dataset/
mv README.txt lspet_dataset/

# Get Annotations
wget http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz
tar zxvf mpii_human_pose_v1_u12_1.tar.gz
rm -rf mpii_human_pose_v1_u12_1.tar.gz
mv mpii_human_pose_v1_u12_1 mpii

# Get Images
cd mpii
wget http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
tar zxvf mpii_human_pose_v1.tar.gz
rm -rf mpii_human_pose_v1.tar.gz

cd ..
