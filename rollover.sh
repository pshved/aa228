#!/bin/bash

RUNS=10

VP=1.5
RV=10
FRA=10000
DEPTH=20

python-jl solve.py simulate --log_every=1 --obstacle_y=-6 --obstacle_policy=obstacle_stay_off.policy --num_runs=$RUNS  --detection_sigma=0.2 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=$FRA --reward_velocity=$RV --velocity_power=$VP --max_depth=$DEPTH --log_file=figure_1_straight_line.csv > figure_1_straight_line.txt &

python-jl solve.py simulate --log_every=1 --obstacle_y=0 --obstacle_policy=obstacle_runover_7.policy --num_runs=$RUNS  --detection_sigma=0.5 --policy=monte-carlo-jl --max_x=30 --num_iterations=300 --future_reward_amplification=$FRA --reward_velocity=$RV --velocity_power=$VP --max_depth=$DEPTH --log_file=figure_1_obstacle.csv > figure_1_obstacle_pullover.txt &

wait


