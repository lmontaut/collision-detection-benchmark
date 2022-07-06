#!/bin/bash

# CONTINUOUS ELLIPSOIDS
for dist in distant close_proximity overlapping
# for dist in distant
do
    for seed in 0 1 2 3
    do
        python exp/continuous_ellipsoids/ellipsoids_benchmark.py --distance_category $dist --num_pairs 250 --seed $seed
    done
done

python plot_exp/continuous_ellipsoids/continuous_ellipsoids_compute_metrics.py
