#!/bin/bash

conda activate nerfstudio

# 指定训练任务的目录
training_script_directory="./"


for ((i=300; i<400; i=i+10))
do
    mkdir ./logs/log_dino_400/dump_epoch${i}_novel_nocd
    python test.py --camera realsense --dump_dir logs/log_dino_400/dump_epoch${i}_novel_nocd --checkpoint_path logs/log_dino_400/minkuresunet_epoch${i}.tar --batch_size 1 --dataset_root ./data3/graspnet --infer --eval --collision_thresh -1 > ./logs/log_dino_400/dump_epoch${i}_novel_nocd/eval.log &
    wait
done