#!/bin/bash

RUNS=10
VP=1
RV=1
FRA=10000

for vp in 0.1 0.5 1 1.5 2; do
	 python-jl solve.py simulate --log_every=1 --obstacle_y=-6 --obstacle_policy=obstacle_stay.policy --num_runs=$RUNS  --detection_sigma=0.1 --policy=monte-carlo-jl --max_x=100 --num_iterations=300 --future_reward_amplification=10 --reward_velocity=10 --velocity_power=$vp > final_velpo_$vp.txt &
done

wait

# Straight line, reward amplification study.
#for ra in 10 100 1000 10000 100000 ; do
for ra in 10 100 1000 10000 100000 ; do
	 python-jl solve.py simulate --log_every=1 --obstacle_y=-6 --obstacle_policy=obstacle_stay.policy --num_runs=$RUNS  --detection_sigma=0.1 --policy=monte-carlo-jl --max_x=20 --num_iterations=300 --future_reward_amplification=$ra > final_amplification_$ra.txt &
done

wait
