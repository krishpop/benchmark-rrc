#!/bin/bash

for ki in .05 0.1 0.15 0.2 0.25
do
    for int in 5 10 15 20
    do
        python grasp_test.py --ki $ki --int_freq $int --use_cp --logdir output/ki$ki-int$int/
    done
done
