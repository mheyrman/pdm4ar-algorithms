from re import I
from typing import Tuple
from xmlrpc.client import Boolean
from matplotlib.pyplot import grid

import numpy as np
from pyrsistent import b
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, State, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function

    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        def check_in_bound(x, y, next_state) -> Boolean:
            next_x, next_y = next_state
            if next_x < 0 or next_y < 0 or next_x >= x or next_y >= y:
                return False
            return True

        def find_start(grid_mdp: GridMdp) -> State:
            n, m = np.shape(grid_mdp.grid)

            x_coords = np.arange(n)
            y_coords = np.arange(m)

            for x in x_coords:
                for y in y_coords:
                    if grid_mdp.grid[(x,y)] == 1:
                        return (x, y)
            return (0,0)

        x, y = np.shape(grid_mdp.grid)

        start = find_start(grid_mdp)
        
        cont = True

        while cont:
            delt = 0
            x_coords = np.arange(x)
            y_coords = np.arange(y)

            future_values = np.zeros_like(grid_mdp.grid).astype(float)

            # for each state s:
            for cur_x in x_coords:
                for cur_y in y_coords: 
                    s = (cur_x, cur_y)

                    utilities = np.ones(5) * -(np.inf-1) # to maximize over all actions
                    # list of s'
                    next_states = [(cur_x - 1, cur_y), (cur_x, cur_y - 1), (cur_x + 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y)]

                    # for each a
                    # NORTH
                    if cur_x - 1 >= 0:
                        a = 0

                        # SUM
                        pos_utility = np.zeros(5)
                        # over each possible s'
                        reward_cur = grid_mdp.stage_reward(s, a)
                        for i in range(0, np.size(pos_utility)):
                            next_state = next_states[i] # extract s'
                            # T(s-a->s') * [R(s-a->s') + gamma * V_last[s']]
                            if check_in_bound(x, y, next_state):
                                # probability of going from s-a->s' *
                                # reward for going to that state + gamma * value of that state
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[next_state])
                            else: # parachute back to start
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[0] = np.sum(pos_utility)

                    # WEST
                    if cur_y - 1 >= 0: # can go west
                        a = 1
                        pos_utility = np.zeros(5)
                        reward_cur = grid_mdp.stage_reward(s, a)
                        for i in range(0, np.size(pos_utility)):
                            next_state = next_states[i]
                            if check_in_bound(x, y, next_state):
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                            else: # parachute back to start
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[1] = np.sum(pos_utility)

                    # SOUTH
                    if cur_x + 1 < x: # can go south
                        a = 2
                        pos_utility = np.zeros(5)
                        reward_cur = grid_mdp.stage_reward(s, a)
                        for i in range(0, np.size(pos_utility)):
                            next_state = next_states[i]
                            if check_in_bound(x, y, next_state):
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                            else: # parachute back to start
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[2] = np.sum(pos_utility) 

                    # EAST           
                    if cur_y + 1 < y:
                        a = 3
                        pos_utility = np.zeros(5)
                        reward_cur = grid_mdp.stage_reward(s, a)
                        for i in range(0, np.size(pos_utility)):
                            next_state = next_states[i]
                            if check_in_bound(x, y, next_state):
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                            else: # parachute back to start
                                pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[3] = np.sum(pos_utility)
                        # this should only give something at goal

                    # STAY
                    if grid_mdp.grid[s] == 0:
                        a = 4
                        pos_utility = grid_mdp.get_transition_prob(s, a, s) * (grid_mdp.stage_reward(s, a) + grid_mdp.gamma * value_func[s])
                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[4] = np.sum(pos_utility)

                    future_values[s] = np.max(utilities)

                    delt = max(delt, np.abs(future_values[s] - value_func[s]))

            if delt < 0.0001:
                cont = False

            value_func=future_values

        # MUST DO THIS
        # for s in states:
        #   opt_policy[s] = argmax_a(sum_s'(P(next_state | state, action) * v[next_state]))
        # return opt_policy
        
        x_coords = np.arange(x)
        y_coords = np.arange(y)

        for cur_x in x_coords:
            for cur_y in y_coords: 
                s = (cur_x, cur_y)

                utilities = np.ones(5) * -(np.inf-1) # to maximize over all actions
                actions = [0, 1, 2, 3, 4]
                # list of s'
                next_states = [(cur_x - 1, cur_y), (cur_x, cur_y - 1), (cur_x + 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y)]

                # for each a
                # NORTH
                if cur_x - 1 >= 0:
                    a = actions[0]
                    # SUM
                    pos_utility = np.zeros(5)
                    # over each possible s'
                    reward_cur = grid_mdp.stage_reward(s, a)
                    for i in range(0, np.size(pos_utility)):
                        next_state = next_states[i] # extract s'
                        # T(s-a->s') * [R(s-a->s') + gamma * V_last[s']]
                        if check_in_bound(x, y, next_state):
                            # probability of going from s-a->s' *
                            # reward for going to that state + gamma * value of that state
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[next_state])
                        else: # parachute back to start
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                    # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                    utilities[0] = np.sum(pos_utility)

                # WEST
                if cur_y - 1 >= 0: # can go west
                    a = actions[1]

                    pos_utility = np.zeros(5)
                    reward_cur = grid_mdp.stage_reward(s, a)
                    for i in range(0, np.size(pos_utility)):
                        next_state = next_states[i]
                        if check_in_bound(x, y, next_state):
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                        else: # parachute back to start
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                    # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                    utilities[1] = np.sum(pos_utility)

                # SOUTH
                if cur_x + 1 < x: # can go south
                    a = actions[2]
                     
                    pos_utility = np.zeros(5)
                    reward_cur = grid_mdp.stage_reward(s, a)
                    for i in range(0, np.size(pos_utility)):
                        next_state = next_states[i]
                        if check_in_bound(x, y, next_state):
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                        else: # parachute back to start
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                    # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                    utilities[2] = np.sum(pos_utility) 

                # EAST           
                if cur_y + 1 < y:
                    a = actions[3]
                    pos_utility = np.zeros(5)
                    reward_cur = grid_mdp.stage_reward(s, a)
                    for i in range(0, np.size(pos_utility)):
                        next_state = next_states[i]
                        if check_in_bound(x, y, next_state):
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma *value_func[next_state])
                        else: # parachute back to start
                            pos_utility[i] = grid_mdp.get_transition_prob(s, a, next_state) * (reward_cur + grid_mdp.gamma * value_func[start])

                    # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                    utilities[3] = np.sum(pos_utility)
                    # this should only give something at goal

                # STAY
                if grid_mdp.grid[s] == 0:
                    a = actions[4]
                    pos_utility = grid_mdp.get_transition_prob(s, a, s) * (grid_mdp.stage_reward(s, a) + grid_mdp.gamma * value_func[s])
    
                    # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                    utilities[4] = np.sum(pos_utility)

                max_index = np.argmax(utilities)
                policy[s] = actions[max_index]
                
        return value_func, policy