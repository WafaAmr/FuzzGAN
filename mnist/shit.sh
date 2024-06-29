#!/bin/bash

# root_path="/home/upc/Desktop/FuzzGAN/mnist/inv_2-red"
# for digit_class in $(ls -d $root_path/*/); do
#     for file in $(ls $digit_class); do
#         if [[ $file == *.npy ]]; then
#             w_path="$digit_class$file"
#             echo $w_path
#             /home/upc/miniconda3/envs/zero/bin/python mnist/shit.py $w_path
#         fi
#     done
# done

root_path="/home/upc/Desktop/FuzzGAN/mnist/inv_2/5/"
for file in $(ls $root_path); do
    if [[ $file == *.npy ]]; then
        w_path="$root_path$file"
        echo $w_path
        /home/upc/miniconda3/envs/zero/bin/python mnist/shit.py $w_path
    fi
done