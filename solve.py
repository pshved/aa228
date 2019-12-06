"""Produce policy for a list of observations.

Example usage (simulation:

python solve.py simulate --policy_file=dummy_without_speeding.policy --log_every=100 --obstacle_y=0 
python solve.py simulate --policy_file=dummy_without_speeding.policy --log_every=100 --obstacle_y=0 --obstacle_policy=obstacle_runover_dummy.policy

"""

from abc import abstractmethod
import bisect
import csv
from collections import defaultdict
import dill as pickle
import math
import os
from os import path
import random
import re
import shutil
import sys

from time import localtime, strftime
from timeit import default_timer as timer

import fire
import pandas as pd

from itertools import product
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
import numpy as np
import scipy as sp
import scipy.stats
from scipy.special import loggamma

from joblib import Parallel, delayed

from tqdm import tqdm

import pandas as pd


print("Loading Julia (this might take a bit)...")

# Julia imports
from julia.api import Julia
import julia
j = julia.Julia()
j.eval('@everywhere push!(LOAD_PATH,"/home/aa228/homework/AA228Student/final-project/GenerativePOMDP.jl/src/")')
import julia.Base
from julia import follow
from julia.POMCPOW import POMCPOWSolver, MaxUCB, MaxQ, MaxTries
from julia.POMDPModelTools import action_info
import julia.POMDPs as POMDPs

class World(object):
    def __init__(self, max_time, max_x, time_increment=0.01):
        self.MAX_TIME = max_time
        self.MAX_X = max_x
        self.time_increment = time_increment
        np.random.seed(333336)

class ObservationBase(object):
    def __init__(self):
        # Know thyself.
        self.agent_x = None
        self.obstacle_y = None

    def julify(self):
        return (self.agent_x, self.obstacle_y)

    @classmethod
    def pythonify(cls, jl_obj):
        self = ObservationBase()
        self.agent_x, self.obstacle_y = jl_obj
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "agent_x={:.3f} o_y={:.3f}".format(self.agent_x, self.obstacle_y)


class Observation(ObservationBase):
    def __init__(self, agent_x, obstacle_y, state, obstacle_x, agent_vx=0.0, agent_ax=0.0):
        self.obstacle_y = obstacle_y
        self.obstacle_x = obstacle_x
        self.agent_x = agent_x
        self.agent_vx = agent_vx
        self.agent_ax = agent_ax
        self.state = state

    def julify(self):
        return (self.agent_x, self.obstacle_y, self.agent_vx, self.agent_ax, self.obstacle_x)

    @classmethod
    def pythonify(cls, jl_obj):
        self = ObservationBase()
        self.agent_x, self.obstacle_y, self.agent_vx, self.agent_ax, self.obstacle_x = jl_obj
        return self

    def b(self):
        """Function approximation for the state."""
        return []

class Scenario(object):
    """What the world is going to do."""
    def __init__(self, obstacle_x=0.0, obstacle_y=0.0, obstacle_h=(2.5*1.75), obstacle_policy=None):

        # Constants
        self.obstacle_pullout_time = 3
        self.obstacle_h = obstacle_h

        self.last_obstacle_action = None
        self.different_obstacle_action = None

        p = Policy(first_key='t',  second_key='t', jitter_keys=['jitter_t', 'jitter_y'])
        if obstacle_policy is None:
            # Stay at the same place all the time
            # We really only need one key...
            p.from_list([
                {'t': -1000, 'action': 'stay'}
            ])
        else:
            if isinstance(obstacle_policy, str):
                p.load(obstacle_policy)
            else:
                p = obstacle_policy
        self.obstacle_policy = p

        # Now that we can sample from the policy, sample the obstacle y.
        self.obstacle_x = obstacle_x
        query = ObservationBase()
        query.agent_x = 0.0
        query.obstacle_y = 0.0
        # Note the use of None for the belief: the adversary assumes perfect observability.
        self.obstacle_y = self.obstacle_policy.sample(query, None, keys='y')
        if self.obstacle_y is None:
            self.obstacle_y = float(obstacle_y)

        # Toyota Camry accelerates to 100 kph in 7 seconds
        self.AX_1 = 100.0 / 3.6 / 7

    def action_velocity(self, action):
        return {
                'stay': (0, 0),
                'pull out': (self.obstacle_h / self.obstacle_pullout_time, +0.5),
                # Make obstacle slow
                'maintain': (3,    0.0), 
                #'maintain': (10,    0.0), 
                }[action]

    def obstacle_action(self, t):
        query = ObservationBase()
        query.agent_x = t
        query.obstacle_y = t
        
        new_action = self.obstacle_policy.sample(query, None, keys='action')
        self.different_obstacle_action = new_action != self.last_obstacle_action
        self.last_obstacle_action = new_action
        return new_action

    def obstacle_movement(self, t):
        return self.action_velocity(self.obstacle_action(t))

class ModeledState(object):
    def __init__(self, agent_x, obstacle_x, obstacle_y):
        self.agent_x = agent_x
        self.obstacle_x = obstacle_x
        self.obstacle_y = obstacle_y

    def julify(self):
        return (self.agent_x, self.obstacle_x, self.obstacle_y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.agent_x is None:
            return "MS oy={:.2f}".format(self.obstacle_y)
        else:
            return "MS x={} oy={:.2f} ox={:.2f}".format(self.agent_x, self.obstacle_y, self.obstacle_x)


    def actions(self):
        return ["maintain", "accelerate", "brake", "slide left", "slide right"]

    def copy(self):
        return ModeledState(self.agent_x, self.obstacle_x, self.obstacle_y)


class State(ModeledState):
    def copy(self):
        c = State(self.world, self.scenario)
        c.__dict__.update(self.__dict__)
        return c


    def __init__(self, world, scenario=None, initial_x=None):
        self.t = 0.0

        self.world = world

        # Obstacle
        self.scenario = scenario
        self.obstacle_x = scenario.obstacle_x
        self.obstacle_y = scenario.obstacle_y
        self.obstacle_h = scenario.obstacle_h

        # Agent
        if initial_x is None and world:
            initial_x = -world.MAX_X
        self.agent_x = float(initial_x)
        self.agent_y = 0.0
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.agent_ax = 0.0
        self.agent_ay = 0.0

        # Mission
        self.SUCCESS_X = world.MAX_X
        # Speed limit is 35 mph
        self.SPEED_LIMIT = 35 * 1609 / 60 / 60 # = 15.6
        # Toyota Camry accelerates to 100 kph in 7 seconds
        self.AX_1 = 100.0 / 3.6 / 7

        # Rewards
        self.REWARD_TIME_PASSAGE = -0.01
        # Article: this reward is not sampleable
        #self.REWARD_MISSION_SUCCESS = 1
        self.REWARD_MISSION_SUCCESS = 10000
        self.REWARD_SPEEDING = -1
        self.REWARD_SLIDE = -2
        # Restrict action "slide right" if the hero is already to the right.
        #self.REWARD_POINTLESS_LATERAL = -2000
        self.REWARD_POINTLESS_LATERAL = 0
        self.REWARD_LATERAL = 0
        #self.REWARD_POINTLESS_LATERAL = -2000
        #self.REWARD_LATERAL = -200
        #self.REWARD_COLLISION = -1000000
        self.REWARD_COLLISION = -100000000
        self.assigned_final_reward = False

        self.PROB_SLIDE_LEFT = 1#0.25

        self.attached_reward = -314.1592

        # self.scenario = {
            # 'obstacle_launch': 5,
            # 'obstacle_sensible': True

    def julify(self, ms=None):
        # Modify states according to the override
        ax = self.agent_x
        oy = self.obstacle_y
        ox = self.obstacle_x
        if ms is not None:
            if ms.agent_x is not None:
                ax = ms.agent_x
            #ox = ms.obstacle_x
            oy = ms.obstacle_y
        return (self.t, ax, float(self.agent_y), self.agent_vx, self.agent_vy, self.agent_ax, self.agent_ay, ox, float(oy), self.obstacle_h, self.attached_reward, self.assigned_final_reward)

    @classmethod
    def pythonize(cls, jl_obj, sc, w):
        self = cls(w, sc)
        self.t, self.agent_x, self.agent_y, self.agent_vx, self.agent_vy, self.agent_ax, self.agent_ay, self.obstacle_x, self.obstacle_y, self.obstacle_h, self.attached_reward, self.assigned_final_reward = jl_obj
        return self

    def attach_reward(self, r):
        self.attached_reward = r

    def isterminal(self):
        return self.assigned_final_reward

    def __str__(self):
        return "x={:.2f} v_x={:.2f} a_x={:.2f} ox={:.2f} oy={:.2f} y={:.2f}".format(self.agent_x, self.agent_vx, self.agent_ax, self.obstacle_x, self.obstacle_y, self.agent_y)

    def str_belief_simple(self):
        return "oy={:.2f}".format(self.obstacle_y)

    def advance(self, dt, action, debug=0):

        reward = 0.0

        if action == 'accelerate':
            self.agent_ax = self.AX_1
        elif action == 'brake':
            self.agent_ax = -self.AX_1
        else:
            self.agent_ax = 0.0
            self.agent_ay = 0.0

        if action == 'slide_left':
            if self.agent_y == 0.5:
                reward += self.REWARD_POINTLESS_LATERAL

            if np.random.rand() < self.PROB_SLIDE_LEFT:
                self.agent_y = 0.5
                reward += self.REWARD_LATERAL 
        elif action == 'slide_right':
            if self.agent_y == 0.0:
                reward += self.REWARD_POINTLESS_LATERAL
            reward += self.REWARD_LATERAL 
            self.agent_y = 0.0

        # Disallow backwards move.
        nvx = max(0.0, self.agent_vx + dt * self.agent_ax)
        nvy = self.agent_vy + dt * self.agent_ay
        
        nx = self.agent_x + dt * nvx
        ny = self.agent_y + dt * nvy

        ovx, ovy = self.scenario.obstacle_movement(self.t)
        if self.scenario.different_obstacle_action and debug > 1:
            print("OMG!  Obstacle is now {} at t={:.2f}".format(self.scenario.last_obstacle_action, self.t))
            print("Policy: {}".format(self.scenario.obstacle_policy))
        
        nox = self.obstacle_x + dt * ovx
        noy = self.obstacle_y + dt * ovy

        if nx > 0 and debug > 0:
            # import ipdb; ipdb.set_trace()
            pass

        #x, y, nx, ny = self.transition_model.act(

        collision = self.intersects(
                self.agent_x, self.agent_y, nx, ny,
                self.obstacle_x, self.obstacle_y, nox, noy,
                self.obstacle_h)

        self.agent_x = nx
        self.agent_y = ny
        self.agent_vx = nvx
        self.agent_vy = nvy
        self.obstacle_x = nox
        self.obstacle_y = noy
        self.t += dt

        if self.agent_y > 0.1:
            reward += self.REWARD_SLIDE

        keep_going = True

        if self.agent_x >= self.SUCCESS_X:
            self.assigned_final_reward = True
            keep_going = False
            reward += self.REWARD_MISSION_SUCCESS

        # Compute agent's trajectory and obstacle's trajectory for the next
        # step so we can detect collisions.  A collision occurs when the
        # trajectories intersect.
        if collision:
            if debug > -1:
                print("Collision at t={} x={}!".format(self.t, self.agent_x))
            reward += self.REWARD_COLLISION
            keep_going = False

        if self.agent_vx > self.SPEED_LIMIT:
            reward += self.REWARD_SPEEDING
        
        # Check if we've accomplished the mission
        if not self.assigned_final_reward and not self.within_bounds(self.world):
            self.assigned_final_reward = True
            keep_going = False

        reward += self.REWARD_TIME_PASSAGE
        return keep_going, reward

    def within_bounds(self, world : World):
        return self.t < world.MAX_TIME and self.agent_x < world.MAX_X


    def intersects(self, x1, y1, x2, y2, ox1, oy1, ox2, oy2, oh):
        """Whether the agent moving from 1 to 2 intersects with the obstacle."""
        critical_x1 = max(x1, ox1)
        critical_x2 = min(x2, ox2)

        # Check if the agent is ahead of behind the obstacle at all times
        if critical_x1 > critical_x2:
            return False
        
        def trisect(a, l, r):
            if a < l: return -1
            if l <= a and a < r: return 0
            return 1

        def interpolate(x1, y1, x2, y2, x):
            if x1 == x2:
                assert x == x1, "x1 == x2 ({}) but x is {}".format(x1, x)
                return y1
            return y1 + (y2-y1) / (x2 - x1) * (x - x1)

        # compute positions of the agent and the obstacle at the critical points
        critical_y1 = interpolate(x1, y1, x2, y2, critical_x1)
        critical_oy1 = interpolate(ox1, oy1, ox2, oy2, critical_x1)
        critical_y2 = interpolate(x1, y1, x2, y2, critical_x2)
        critical_oy2 = interpolate(ox1, oy1, ox2, oy2, critical_x2)

        # If the agent stayed on one side of obstacle in these critical points,
        # it didn't collide.
        side1 = trisect(critical_y1, critical_oy1 - oh, critical_oy1)
        side2 = trisect(critical_y2, critical_oy2 - oh, critical_oy2)
        return abs(side1 + side2) != 2



class ObservationModel(object):
    def __init__(self,
            obstacle_y_sigma=0.0,
            precision=0.1,
            adaptive_detection_factor=1.0,
            adaptive_detection_distance=5.0):
        self.precision = precision
        self.obstacle_y_sigma = obstacle_y_sigma
        self.adaptive_detection_factor = adaptive_detection_factor
        self.adaptive_detection_distance = adaptive_detection_distance

    def detection_sigma(self, state):
        if state.agent_x < state.obstacle_x - self.adaptive_detection_distance:
            return self.adaptive_detection_factor * self.obstacle_y_sigma
        else:
            return self.obstacle_y_sigma

    def observe(self, state, jitter=True):
        """Observe state and modify its parameters according to the observation model.
        
        state can be state or observation
        """
        r = state.obstacle_y
        if jitter:
            r = np.random.normal(r, self.detection_sigma(state))
        state.obstacle_y = self.clip(r)
        return state

    def clip(self, r):
        return math.trunc(r / self.precision) * self.precision

    def O(self, o, s, a):
        """Probability O(o | s,a) of observing o after taking a in state s."""
        return self.precision * sp.stats.norm.pdf(o.obstacle_y, s.obstacle_y, self.detection_sigma(s)) 

    def sample(self, state, action):
        o = Observation(
                agent_x=state.agent_x,
                obstacle_y=state.obstacle_y,
                obstacle_x=state.obstacle_x,
                state=state)

        return self.observe(o)

# Note: in practice, we can create a transition model form observations in the field.
class TransitionModel(object):
    def __init__(self, world, initial_obstacle_y=0, om=None):
        self.initial_obstacle_y = initial_obstacle_y
        self.om = om
        self.world = world

    def sample_from_initial_belief(self, initial_state):
        state = initial_state.copy()
        #w = 2*self.om.detection_sigma(initial_state)
        w = 5
        state.obstacle_y = np.random.uniform(
                    low=self.initial_obstacle_y - w,
                    high=self.initial_obstacle_y + w)
        return self.om.observe(state)

    def sample(self, state, action):
        # For now, let's assume that our action doesn't affect the obstacle action.
        ns = state.copy()
        # Advance to the next state as is (all the randomness is already included)
        ns.advance(self.world.time_increment, action)

        # Reset the obstacle y.
        return self.om.observe(ns)


class Belief(object):
    def with_state(self, s):
        self.state = s
        return self

    def julify(self):
        return [self.state.julify(ms=b) for b in self.Bs]

    def __init__(self):
        self.Bs = []

    def add_initial_belief(self, state, G, num_samples=1):
        """Adds the samples from the generative model to the existing belief."""
        for _ in range(num_samples):
            self.Bs.append(G.sample_from_initial_belief(state))

    def particles(self):
        return self.Bs

    def sample(self):
        """Returns a random sample from the distribution.

        By the definition of the particle filter, this is a uniformly
        distributed particle."""
        return self.Bs[np.random.choice(np.arange(len(self.Bs)))]


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} particles, {}".format(len(self.Bs), self.Bs)

    def simple(self):
        return "{} particles, {}".format(len(self.Bs), list(map(lambda b: b.str_belief_simple(), self.Bs)))

    def mixin_particles(self, s, a, o, O, G, new_particles=None, mixin=0.5):
        N = len(self.Bs)
        for i in range(N):
            self.Bs.append(G.sample(s, a))
        self.update_particles(a, o, O, G, cutoff=N)

    def update_particles(self, a, o, O, G, new_particles=None, cutoff=None):
        if cutoff is None:
            cutoff = len(self.particles())
        # States and weights
        ss = []
        ws = []
        for si in self.particles():
            # s_i is of type Observation
            sj = G.sample(si, a)
            wi = O.O(o, sj, a)
            ss.append(sj)
            ws.append(wi)
        N = len(self.Bs)
        new_Bs = []
        if len(ss) == 0: return

        # Compute probabilities of each particle
        p = np.array(ws)
        ps = np.sum(p)
        if ps < 0.000001:
            p = np.ones_like(p) / p.shape[0]
        else:
            p = ws / ps
        #print("sample input: ss={}, ws={}".format(ss, p))

        if new_particles is None:
            new_particles = N
        for _ in range(cutoff):
            try:
                new_Bs.append(ss[np.random.choice(np.arange(N), p=p)])
            except Exception as e:
                print("OOPS: ss={}, ws={}".format(ss, ws))

        #print("sample output: bs={}".format(new_Bs))
        self.Bs = new_Bs

    def function_approximation(self, om: ObservationModel, w : World):
        """Approximates the belief state into something we can build alpha-vectors for.
        
        Returns the following components:
        TODO.  Seems promising: distance to the obstacle clipped to some sensible region (0;100m).
        Probability of obstacle below 0.
        relative speed.
        
        """
        # TODO.
        return []

class HashDict(dict):
    def __init__(self, fn, *args, **kwargs):
        self._hash_fn = fn
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, val):
        key = fn(key)
        return super().__setitem__(key, val)

    def __getitem__(self, key):
        key = fn(key)
        return super().__getitem__(key)

class MonteCarloTreeSearchAlgorithm(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.Q = defaultdict(lambda: HashDict(lambda h: tuple([str(hi) for hi in h])))

    def select_action(self, b : Belief, t, budget=100):
        """t is time horizon."""
        h = []
        for _ in range(budget):
            s = b.sample()
            self.simulate(s, h, t)

        qas = []
        for a in self.Q:
            qas.append((max(self.Q[a].values()), a))

        return max(qas)[0]

    def simulate(self, s, h, t):
        # Dummy simulation.
        for a in s.actions():
            self.Q[a][h] = 0

class JuliaLoader(object):
    def __init__(self):
        pass


def estimate_value_fn(pomdp, s, h_BeliefNode, steps):
    import ipdb; ipdb.set_trace()
    print(s)


class POMDPForJulia(JuliaLoader):
    def __init__(self):
        super().__init__()
        self.tree_in_info = False
        self.max_depth = 20

    def init_pomdp(self, gamma):
        self.pomdp = follow.MakeGenerativePOMDP(
                [self.random_state()],
                self.actions(),
                [self.random_observation()],
                self.transition,
                self.observation,
                self.reward,
                self.initial_state_prior,
                self.observation_weight,
                self.estimate_value,
                gamma)

    def init_solver(self, fixup='brake'):
        self.solver = follow.fixup_estimate(POMCPOWSolver(
                max_depth=self.max_depth,
                enable_action_pw=False,
                # Our rewards are small, so we need to tone this down.
                criterion=MaxUCB(10),
                #criterion=MaxQ(),
                check_repeat_obs=True,
                check_repeat_act=False,
                tree_in_info=self.tree_in_info,
                # Max 60 seconds per iteration.
                max_time=60,
                #estimate_value=0,
        ))
        if fixup == 'brake':
            self.solver = follow.fixup_estimate(self.solver)
        else:
            self.solver = follow.fixup_estimate_maintain(self.solver)

    def select_action(self, belief, observation):
        fixup = 'brake'
        if belief.state.agent_x > -2:
            fixup = 'maintain'
        self.init_solver(fixup)
        self.policy = POMDPs.solve(self.solver, self.pomdp)
        if self.tree_in_info:
            #a, info = POMDPs.action_info(self.policy, belief.julify())
            a, info = action_info(self.policy, belief.julify())
        else:
            a = POMDPs.action(self.policy, belief.julify())
        return a

    @abstractmethod
    def actions(self):
        print("CALLING actions")

    @abstractmethod
    def random_state(self):
        print("CALLING random_state")

    @abstractmethod
    def random_observation(self):
        print("CALLING random_observation")

    @abstractmethod
    def initial_state_prior(self):
        print("CALLING initial_state_prior")

    @abstractmethod
    def transition(self, s, a):
        print("CALLING transition")

    @abstractmethod
    def observation(self, s, a, sp):
        print("CALLING observation")

    @abstractmethod
    def observation_weight(self, s, a, sp, o):
        print("CALLING observation_weight")

    @abstractmethod
    def reward(self, s, a, sp, o):
        print("CALLING reward")

    @abstractmethod
    def estimate_value(self, s):
        print("CALLING estimate_value")
        assert False


class MonteCarloJulia(POMDPForJulia):
    def __init__(self, gamma, scenario, world, observation_model,
            future_reward_amplification=1,
            reward_velocity=100,
            velocity_power=2,
            max_depth=20):
        self.scenario = scenario
        self.world = world
        self.initial_state = State(self.world, self.scenario)
        self.observation_model = observation_model


        # Load Julia before.
        super().__init__()

        self.tree_in_info = False
        self.max_depth=max_depth

        self.init_pomdp(gamma)
        self.init_solver()

        #self.FUTURE_REWARD_AMPLIFICATION = future_reward_amplification
        self.FUTURE_REWARD_AMPLIFICATION = self.initial_state.REWARD_MISSION_SUCCESS
        self.REWARD_VELOCITY = reward_velocity
        self.velocity_power = velocity_power


    def actions(self):
        return [
            'maintain', 
            'accelerate', 
            'brake', 
            'slide_left', 
            'slide_right']

    def random_state(self):
        s = self.initial_state.julify()
        print("CALLED: Random State Prior --> {} ({})".format(self.initial_state, s))
        return s

    def random_observation(self):
        return Observation(agent_x=0.0, obstacle_y=-100.0, state=None, obstacle_x=0.0).julify()

    def initial_state_prior(self):
        print("CALLED: Initial State Prior")

    def transition(self, s, a):
        #print("CALLED: Transition")
        state = State.pythonize(s, self.scenario, self.world)
        keep_iterating, r = state.advance(self.world.time_increment, a)

        # TODO: do something if keep_iterating is false
        #assert keep_iterating, "TODO!"

        state.attach_reward(r)

        return state.julify()

    def observation(self, s, a, sp):
        #print("CALLED: observation")
        sp_ = State.pythonize(sp, self.scenario, self.world)
        o = self.observation_model.sample(sp_, a)
        return o.julify()

    def observation_weight(self, s, a, sp, o):
        sp_ = State.pythonize(sp, self.scenario, self.world)
        o_ = Observation.pythonify(o)
        w = self.observation_model.O(o_, sp_, a)
        return w

    def SpeedReward(self, s: State):
        r = s.agent_vx
        r = min(max(r, 0), s.SPEED_LIMIT)
        return math.pow(r, self.velocity_power) * self.REWARD_VELOCITY


    def FutureReward(self, s : State):
        # How long would it take to get to the end with the current velocity
        to_go = float(self.world.MAX_X - s.agent_x)

        went = float(self.world.MAX_X + s.agent_x)

        total = float(self.world.MAX_X + self.world.MAX_X)

        return went / total * self.FUTURE_REWARD_AMPLIFICATION

        vx = s.agent_vx
        # The agent has stopped.
        if vx <= 0.00001:
            # Assume that we start going on the next step...
            vx = s.AX_1 * self.world.time_increment
            # ... but we lose one unit of time.
            to_go += vx * self.world.time_increment * 100

        # ... actually, 1 is too few, let's lose more.
        return - (to_go / vx) / self.world.time_increment * self.FUTURE_REWARD_AMPLIFICATION

        # vx = s.agent_vx + s.agent_ax
        # if vx <= 0.001:
            # vx = float(s.AX_1) / 2
            # to_go *= 1.1
        # 
        # v = (to_go / vx) / self.world.time_increment * s.REWARD_TIME_PASSAGE
        # if abs(v) > 1000:
            # pass
        # return (to_go / vx) / self.world.time_increment * s.REWARD_TIME_PASSAGE

        # Distance to the end
        #return (self.world.MAX_X - s.agent_x) * s.REWARD_TIME_PASSAGE * 100

    def reward(self, s, a, sp, o):
        #print("CALLED: reward")
        sp_ = State.pythonize(sp, self.scenario, self.world)
        s_ = State.pythonize(s, self.scenario, self.world)
        assert sp_.attached_reward is not None

        r = sp_.attached_reward

        # Now let's add the FutureCost heuristic

        dr1 = self.FutureReward(sp_) - self.FutureReward(s_)
        #print("Reward movement {} --> {}, delta={}".format(self.FutureReward(sp_), self.FutureReward(s_), dr))

        r += dr1

        dr2 = self.SpeedReward(sp_)

        r += dr2

        #print("Sampled {} from {} to {}.  Reward {:.4f} from movement {} --> {}, delta={} + {}".format(a, s_, sp_, r, self.FutureReward(s_), self.FutureReward(sp_), dr1, dr2))

        return r

    def estimate_value(self, s):
        s_ = State.pythonize(s, self.scenario, self.world)
        return s_.isterminal()



class OnlinePolicy(object):
    def __init__(self, search_algorithm):
        self.search_algorithm = search_algorithm
        self.horizon = 20

    def sample(self, observation, belief):
        # This actually ignores observation but assumes belief has been updated
        # with the observation.
        return self.search_algorithm.select_action(belief, self.horizon)



class Policy(object):
    def __init__(self, first_key='agent_x', second_key='obstacle_y', jitter_keys=[]):
        self.G = []
        self.first_key = first_key
        self.second_key = second_key
        # Optional jitter
        self.jitter_keys = jitter_keys

    def target(self,jitter_key):
        return re.sub(r'.*_', '', jitter_key)

    def __str__(self):
        return str(self.pi)

    def load(self, policy_file):
        """Produces self.pi.

        self.pi = list( (x, [ (y, dict(row)) ]))
        """
        entries = []
        with open(policy_file, 'r') as f:
            r = csv.DictReader(f)
            #header = next(r, None)
            for row in r:
                for k in row:
                    try:
                        row[k] = float(row[k])
                    except:
                        pass
                entries.append(row)

        return self.from_list(entries)

    def from_list(self, entries):
        # Group by x
        gx = {}
        for e in entries:
            k = e[self.first_key]
            gx[k] = gx.get(k, []) + [e]

        accummulated_jitter = defaultdict(float)

        pi = sorted(gx.items(), key=lambda kv: kv[0])

        pis = []
        for e in pi:
            for jk in self.jitter_keys:
                if jk not in e[1][0]: continue
                jitter_sigma = e[1][0][jk]
                jitter = np.random.normal(0.0, jitter_sigma)
                accummulated_jitter[jk] += jitter

            # Add jitter to all
            def with_jitter(entry, key, jitter):
                if jitter == 0.0: return entry
                if self.target(key) not in entry: return entry
                ee = dict(entry.items())
                ee[self.target(key)] += jitter
                return ee
            def with_jitters(entry, keys, jitters):
                ee = dict(entry.items())
                for key in keys:
                    if self.target(key) not in entry: continue
                    ee[self.target(key)] += jitters[key]
                return ee

            remapped_e1 = map(lambda entry: with_jitters(entry, self.jitter_keys, accummulated_jitter), e[1])

            e0 = with_jitters({self.first_key: e[0]}, self.jitter_keys, accummulated_jitter)[self.first_key]

            pis.append((e0, sorted(remapped_e1, key=lambda v: v[self.second_key])))

        self.pi = pis

        print(self.pi)
        return self

    def sample(self, observation, belief, keys='action'):
        """Sample action

        observation.agent_x is used as the first key
        observation.obstacle_y is used as the second key
        """

        if isinstance(keys, str):
            return self._sample_key(observation, keys)
        return [self._sample_key(observation, k) for k in keys]

    def _sample_key(self, observation : Observation, key):
        best_action = 'maintain'
        #print("Sampling {} from observation {}".format(key, observation))
        # Find the entry in the policy.
        # (this is not super efficient, but leaving this as TODO)
        i = bisect.bisect_right(list(map(lambda v: v[0], self.pi)), observation.agent_x)
        if i == 0:
            return best_action
        pi_x = self.pi[i-1]
        j = bisect.bisect_right(list(map(lambda v: v[self.second_key], pi_x[1])), observation.obstacle_y)
        if j == 0:
            return best_action

        best_action = pi_x[1][j-1].get(key, None)
        return best_action



class Simulator(object):
    def __call__(
            self,
            policy_file=None,
            policy='from_file',
            num_runs=1,
            log_every=100,
            log_file="/dev/null",
            detection_sigma=1.0,
            adaptive_detection_factor=1.0,
            adaptive_detection_distance=5.0,
            obstacle_y=0.0,
            obstacle_policy=None,
            gamma=0.99,
            num_iterations=None,
            max_x=20,
            future_reward_amplification=1,
            reward_velocity=0,
            velocity_power=1,
            max_depth=20):

        logf = open(log_file, "w", newline='') 
        logf_csv = csv.writer(logf, delimiter=",")

        # simulation parameters
        obstacle_length = 4.5
        time_increment = 0.1
        #world = World(300, 20, time_increment)
        world = World(300, max_x, time_increment)

        observation_model = ObservationModel(
                obstacle_y_sigma=detection_sigma,
                adaptive_detection_factor=adaptive_detection_factor,
                adaptive_detection_distance=adaptive_detection_distance)

        transition_model = TransitionModel(
                world, om=observation_model, initial_obstacle_y=obstacle_y)


        # Scores
        scores = pd.DataFrame(columns=['score', 'run'])

        overall_action_counts = b = pd.DataFrame.from_dict({'count': {}})
        timings = []

        # Initial State
        for run in range(num_runs): 
            action_counts = defaultdict(int)
            scenario = Scenario(obstacle_y=obstacle_y, obstacle_policy=obstacle_policy)
            detection_changed = False

            # Load hero's policy
            if policy == 'from_file':
                policy = Policy().load(policy_file)
                print("Policy {}".format(policy.pi))
            elif policy == 'monte-carlo':
                mcts = MonteCarloTreeSearchAlgorithm(gamma)
                policy = OnlinePolicy(mcts)
            elif policy == 'monte-carlo-jl':
                algo = MonteCarloJulia(gamma, scenario, world, observation_model,
                        future_reward_amplification=future_reward_amplification,
                        reward_velocity=reward_velocity,
                        velocity_power=velocity_power,
                        max_depth=max_depth)
                policy = OnlinePolicy(algo)

            state = State(world, scenario)
            reward = 0
            iteration = -1

            keep_iterating = True
            belief = Belief()
            belief.add_initial_belief(state, transition_model, 10)
            # Default action, just to initialize the observations...
            no = observation_model.sample(state, 'maintain')

            while keep_iterating:
                iteration += 1
                # Select action.  Since we need the observation, we save it from the previous run.
                o = no

                t_start = timer()

                a = policy.sample(o, belief.with_state(state))
                no = observation_model.sample(state, a)
                #belief.update_particles(a, no, observation_model, transition_model)
                belief.mixin_particles(state, a, no, observation_model, transition_model, mixin=0.5)
                keep_iterating, r = state.advance(time_increment, a, debug=2)

                t_end = timer()

                timings.append(t_end - t_start)

                if iteration % log_every == 0 or not keep_iterating:
                    print("=====================================")
                    print("It {} t={:.2f}, state {} ; belief {}".format(iteration, state.t, state, belief.simple() ))
                    print("Selected action={} reward={}".format(a, r))
                    print("New Belief {}".format(belief))

                reward += r
                if num_iterations is not None and keep_iterating:
                    keep_iterating = keep_iterating and (iteration < num_iterations)

                # Logging.
                action_counts[a] += 1
                # iteration, state, action, reward, observation
                log_row = []
                log_row += [run, iteration, state.t]
                log_row.append(a)
                log_row.append(r)
                log_row.append(int(keep_iterating))
                log_row += list(state.julify())
                log_row += list(no.julify())
                log_row += list([list(x) for x in belief.julify()])
                logf_csv.writerow(log_row)
                logf.flush()


            # Tally scores
            print("Simulation done!  Reward = {}".format(reward))
            scores = scores.append(pd.DataFrame([{'run': run, 'score': reward}], index=['run']))

            # Tally actions
            actions = pd.DataFrame.from_dict({'count': action_counts})
            print("Action counts\n{}".format(actions.sort_values(by='count', ascending=False)))
            overall_action_counts = overall_action_counts.add(actions, fill_value=0)


        print("\n\n\n============================");
        print("\tFinal Report:\n\n")
        hist, edges = np.histogram(np.array(timings))
        timings_df = pd.DataFrame.from_dict({'bin': edges[1:], 'timings': hist}).sort_values(by='bin')
        print("Solution Timings:\n{}".format(timings_df))
        actions_hist = overall_action_counts.sort_values(by='count')
        print("Action histogram:\n{}".format(actions_hist))
        avg_score = scores.mean()['score']
        print("Average score after {} runs: {}".format(num_runs, avg_score))

        log_timings = re.sub(r'\.', '_timings.', log_file)
        timings_df.to_csv(log_timings)
        log_actions = re.sub(r'\.', '_actions.', log_file)
        actions_hist.to_csv(log_actions)
        log_score = re.sub(r'\.', '_score.', log_file)
        scores = scores.append(pd.DataFrame([{'run': -1, 'score': avg_score}], index=['run']))
        scores.to_csv(log_score)

        logf.close()


    def foo(self):
        print("Foo")

class SelfDriving(object):
    def __init__(self):
        self.simulate = Simulator()
        self.solve = PolicySolver()


if __name__ == '__main__':
    fire.Fire(SelfDriving)


