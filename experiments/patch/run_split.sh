#!/bin/bash

aff_folder=/home/maisl/workspace/ppp/wormbodies/setup08_191106_00/val/processed/split2
affkey=volumes/pred_affs
output_folder=/home/maisl/workspace/ppp/wormbodies/setup08_191106_00/val/mws_on_dense_affinities

samples=($(ls $aff_folder))

for s in "${samples[@]}";
do
    echo $s
    aff_path=${aff_folder}/${s}
    python experiment.py $aff_path $affkey $output_folder
done

