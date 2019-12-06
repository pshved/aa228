#!/bin/bash

RUNS=10

VP=1
RV=0
FRA=10000

for DEPTH in 10 20 40 80; do
python-jl solve.py simulate --log_every=1 --obstacle_policy=obstacle_always_on.policy --num_runs=$RUNS  --detection_sigma=0.2 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=$FRA --reward_velocity=$RV --velocity_power=$VP --max_depth=$DEPTH > figure_2_stay_on_depth_$DEPTH.txt &
done

python-jl solve.py simulate --log_every=1 --obstacle_policy=obstacle_stay_off.policy --num_runs=$RUNS  --detection_sigma=0.2 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=$FRA --reward_velocity=$RV --velocity_power=$VP --max_depth=20 > figure_1_straight_line.txt &


wait



