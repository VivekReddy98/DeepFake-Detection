#!/bin/bash

SRCPATH="/mnt/beegfs/ppatel27/data/dfdc_train_part_"

for i in {7..15};
  #do mpirun -np 8 ./paralellEncode.py $SRCPATH$i /mnt/beegfs/vkarri/tfrecords/ mp4  ;
done
