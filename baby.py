from julia.QuickPOMDPs import *
from julia.POMDPs import solve, pdf
from julia.QMDP import QMDPSolver
from julia.POMDPSimulators import stepthrough
from julia.POMDPPolicies import alphavectors


#from julia import Main
import julia
j = julia.Julia()
#j.eval("/home/aa228/.julia/config/startup.jl")
j.eval('@everywhere push!(LOAD_PATH,"/home/aa228/homework/AA228Student/final-project/GenerativePOMDP.jl/src/")')
#Main.include("/home/aa228/homework/AA228Student/final-project/follow.jl")
#from julia.GenertaivePOMDP import follow
from julia import follow
from julia.POMCPOW import POMCPOWSolver

S = ['left', 'right']
A = ['left', 'right', 'listen']
O = ['left', 'right']
γ = 0.95

def T(s, a, sp):
    if a == 'listen':
        return s == sp
    else: # a door is opened
        return 0.5 #reset

def Z(a, sp, o):
    if a == 'listen':
        if o == sp:
            return 0.85
        else:
            return 0.15
    else:
        return 0.5

def R(s, a):
    if a == 'listen':
        return -1.0
    elif s == a: # the tiger was found
        return -100.0
    else: # the tiger was escaped
        return 10.0

m = follow.GenerativePOMDP(S,A,O,T,Z,R,γ)

solver = POMCPOWSolver()
policy = solve(solver, m)

print()

rsum = 0.0
for step in stepthrough(m, policy, max_steps=10):
    print('s:', step.s)
    print('a:', step.a)
    print('r:', step.r, '\n')
    print('sp:', step.sp, '\n')
    rsum += step.r

print('Undiscounted reward was', rsum)
