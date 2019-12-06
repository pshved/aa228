#!/bin/bash

RUNS=20

VP=1.5
RV=10
#VP=1
#RV=0
FRA=10000
DEPTH=40

for DEPTH in 10 20 40 80; do
python-jl solve.py simulate --log_every=1 --obstacle_policy=obstacle_blocked.policy --num_runs=$RUNS  --detection_sigma=0.2 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=$FRA --reward_velocity=$RV --velocity_power=$VP --max_depth=$DEPTH --log_file=figure_3_blocked_depth_$DEPTH.csv > figure_3_blocked_depth_$DEPTH.txt &
done

wait
