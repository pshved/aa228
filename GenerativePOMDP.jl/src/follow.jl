module follow

using POMDPModelTools
using POMDPs
using POMCPOW
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using Parameters

using Random

using ParticleFilters

using BeliefUpdaters
using MCTS

import POMDPs: action, value, solve, updater

export
	GenerativePOMDP
	solve


struct GenerativePOMDP{S, A, O, OF, RF, TF, IDF, OW, EV} <: POMDP{S,A,O}
	s::Vector{S}
	a::Vector{A}
	o::Vector{O}

	of::OF
	rf::RF
	# TF/IDF lol
	# (It's "Trainsition Function" / "Initial Distribution Function")
	tf::TF
	idf::IDF
	ow::OW
	ev::EV

	gamma::Float64
end

# We're gonna skip on the RNG, and use it from python
POMDPs.gen(::DDNNode{:sp}, m::GenerativePOMDP, s, a, rng) = m.tf(s, a)
POMDPs.gen(::DDNNode{:o}, m::GenerativePOMDP, s, a, sp, rng) = m.of(s, a, sp)
POMDPs.reward(m::GenerativePOMDP, s, a, sp, o) = m.rf(s, a, sp, o)
POMDPs.initialstate_distribution(m::GenerativePOMDP) = m.idf
POMDPs.discount(m::GenerativePOMDP) = m.gamma
POMDPs.actions(m::GenerativePOMDP) = m.a
ParticleFilters.obs_weight(m::GenerativePOMDP, s, a, sp, o) = m.ow(s, a, sp, o)

#estimate_value(m::GenerativePOMDP, m2::GenerativePOMDP, s, h, steps) = m.ev(s, h, steps)

POMDPs.isterminal(m::GenerativePOMDP, s) = m.ev(s)

"""
Selects specific action
Copied from
    BrakePolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
a generic policy that uses the actions function to create a list of actions and then randomly samples an action from it.

Constructor:

    `BrakePolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `NothingUpdater` in the above constructor)
"""
mutable struct BrakePolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
BrakePolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater()) = BrakePolicy(rng, problem, updater)

## policy execution ##
function action(policy::BrakePolicy, s)
    return "brake"
end

function action(policy::BrakePolicy, b::Nothing)
    return "brake"
end

## convenience functions ##
updater(policy::BrakePolicy) = policy.updater


"""
solver that produces a random policy
"""
mutable struct BrakeSolver <: Solver
    rng::AbstractRNG
end
BrakeSolver(;rng=Random.GLOBAL_RNG) = BrakeSolver(rng)

solve(solver::BrakeSolver, problem::Union{POMDP,MDP}) = BrakePolicy(solver.rng, problem, NothingUpdater())

mutable struct MaintainPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
MaintainPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater()) = MaintainPolicy(rng, problem, updater)

## policy execution ##
function action(policy::MaintainPolicy, s)
    return "maintain"
end

function action(policy::MaintainPolicy, b::Nothing)
    return "maintain"
end

## convenience functions ##
updater(policy::MaintainPolicy) = policy.updater


"""
solver that produces a random policy
"""
mutable struct MaintainSolver <: Solver
    rng::AbstractRNG
end
MaintainSolver(;rng=Random.GLOBAL_RNG) = MaintainSolver(rng)

solve(solver::MaintainSolver, problem::Union{POMDP,MDP}) = MaintainPolicy(solver.rng, problem, NothingUpdater())

function fixup_estimate(m::POMCPOWSolver)
	m.estimate_value = RolloutEstimator(BrakeSolver(m.rng))
	return m
end

function fixup_estimate_maintain(m::POMCPOWSolver)
	m.estimate_value = RolloutEstimator(MaintainSolver(m.rng))
	return m
end

# Note that s, a, and o vectors are only used to infer the type of the state and action element.
function MakeGenerativePOMDP(s, a, o, tf, z, r, idf, ow, ev, discount)
	ss = vec(collect(s))
	as = vec(collect(a))
	os = vec(collect(o))

  ST = eltype(ss)
  AT = eltype(as)
  OT = eltype(os)

	re = RolloutEstimator(BrakeSolver())

	m = GenerativePOMDP(ss, as, os, z, r, tf, idf, ow, ev, discount)

	return m

end


#function POMDPs.gen(m::MyPOMDP, s, a, rng)
#function POMDPs.gen(::DDNNode{:sp}, m::MyPOMDP, s, a, rng)
function TF(s, a)
		rng = Random.MersenneTwister(123)
    # transition model
    if a # feed
        sp = false
    elseif s # hungry
        sp = true
    else # not hungry
        sp = rand(rng) < 0.10
    end
		return sp
end

#function POMDPs.gen(::DDNNode{:o}, m::MyPOMDP, s, a, sp, rng)
function O(s, a, sp)
		rng = Random.MersenneTwister(123)
    # observation model
    if sp # hungry
        o = rand(rng) < 0.60
    else # not hungry
        o = rand(rng) < 0.15
    end
		return o
end

#function POMDPs.reward(m::MyPOMDP, s, a, sp, o)
function R(s, a, sp, o)
		rng = Random.MersenneTwister(123)
    # reward model
    r = s*10 + a*1

    # create and return a NamedTuple
		return r
end

function OW(s, a, sp, o)
		return 1.0
end

# ISF = Deterministic(false)
# 
# solver = POMCPOWSolver(max_depth=1, criterion=MaxUCB(20.0))
# 
# S = [true, false]
# A = [true, false]
# OS = [true, false]
# 
# pomdp = MakeGenerativePOMDP(S, A, OS, TF, O, R, ISF, OW, 0.9)
# 
# planner = solve(solver, pomdp)
# 
# hr = HistoryRecorder(max_steps=100)
# hist = simulate(hr, pomdp, planner)
# for (s, b, a, r, sp, o) in hist
#     @show s, a, r, sp
# end
# 
# rhist = simulate(hr, pomdp, BrakePolicy(pomdp))
# println("""
#     Cumulative Discounted Reward (for 1 simulation)
#         Random: $(discounted_reward(rhist))
#         POMCPOW: $(discounted_reward(hist))
#     """)
# 
# struct MyPOMDP{S, A, O} <: POMDP{S,A,O}
# 	s::Vector{S}
# 	a::Vector{A}
# 	o::Vector{O}
# end
# POMDPs.discount(m::MyPOMDP) = 0.9
# POMDPs.states(m::MyPOMDP) = m.s
# POMDPs.actions(m::MyPOMDP) = m.a

end
