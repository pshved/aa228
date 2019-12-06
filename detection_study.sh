#!/bin/bash

RUNS=50

for adf in 1 2 5 10; do
	python-jl solve.py simulate --log_every=1 --obstacle_policy=obstacle_stay_on.policy --num_runs=$RUNS  --detection_sigma=0.1 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=10 --reward_velocity=10 --velocity_power=1.5 --max_depth=20 --adaptive_detection_factor=$adf --log_file=figure_4_stay_on_adf_${adf}.csv > figure_4_stay_on_adf_${adf}.txt &
done

wait



