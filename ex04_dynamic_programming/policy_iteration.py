from ctypes import util
from typing import Tuple

import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function

class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:

        # todo implement here

        def check_in_bound(x, y, next_state):
            next_x, next_y = next_state
            if next_x < 0 or next_y < 0 or next_x >= x or next_y >= y:
                return False
            return True

        def find_start(grid_mdp: GridMdp):
            n, m = np.shape(grid_mdp.grid)

            x_coords = np.arange(n)
            y_coords = np.arange(m)

            for x in x_coords:
                for y in y_coords:
                    if grid_mdp.grid[(x,y)] == 1:
                        return (x, y)
            return (0,0)
#################################################################################################
        # Policy Iteration:
#################################################################################################
        # stable = false
        # while stable = false:
        #   v(s) = policy evaluation(r(s), p(s'|s,a), gamma, threshold, policy[s], v[s])
        #   policy[s], changed? = policy improvement(p(s'|s,a), policy[s], v[s])
        #   if changed? = false:
        #       stable = true
        # policy*[s] = polcy[s]
        # return policy*[s]
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        start = find_start(grid_mdp)
        x, y = np.shape(grid_mdp.grid)   
        x_coords = np.arange(x)
        y_coords = np.arange(y)

        stable = False

        while not stable:
            value_func = np.zeros_like(grid_mdp.grid).astype(float)
            # POLICY EVALUATION
            cont = True

            while cont:
                delt = 0
                future_values = np.zeros_like(grid_mdp.grid).astype(float)

                for x_cur in x_coords:
                    for y_cur in y_coords: # for every state s
                        s = (x_cur, y_cur)
                        past_value = value_func[s]
                        next_states = [(x_cur - 1, y_cur), (x_cur, y_cur - 1), (x_cur + 1, y_cur), (x_cur, y_cur + 1), (x_cur, y_cur)]

                        utility = 0

                        # sum (s') { T(s-policy(s)->s') * [R(s) + gamma * value(s')] }
                        for next_state in next_states:
                            trans_prob = grid_mdp.get_transition_prob(s, policy[s], next_state)
                            reward = grid_mdp.stage_reward(s, policy[s])
                            if check_in_bound(x, y, next_state):
                                utility += trans_prob * (reward + grid_mdp.gamma * value_func[next_state])
                            else:
                                utility += trans_prob * (reward + grid_mdp.gamma * value_func[start])

                        future_values[s] = utility

                        delt = max(delt, np.abs(past_value - utility))

                value_func = future_values

                if delt < 0.0001:
                    cont = False

            # POLICY IMPROVEMENT
            stable = True

            for x_cur in x_coords:
                for y_cur in y_coords: # for every state s
                    s = (x_cur, y_cur)
                    past_policy = policy[s]
                    next_states = [(x_cur - 1, y_cur), (x_cur, y_cur - 1), (x_cur + 1, y_cur), (x_cur, y_cur + 1), (x_cur, y_cur)]
                    actions = [0, 1, 2, 3, 4] # actions are NORTH, WEST, SOUTH, EAST, STAY

                    utilities = np.ones(5) * -(np.inf)
                    
                    # policy[s] = argmax (a) {sum (s') {T(s-a->s') * [R(s) + gamma * values(s')]}}
                    if check_in_bound(x, y, next_states[0]):   # can go NORTH
                        a = actions[0]                              # go north
                        pos_utility = 0
                        # sum (s') { T(s-policy(s)->s') * [R(s) + gamma * value(s')] }
                        reward = grid_mdp.stage_reward(s, a)
                        for next_state in next_states:
                            trans_prob = grid_mdp.get_transition_prob(s, a, next_state)
                            if check_in_bound(x, y, next_state):
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[next_state])
                            else:
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[start])
                        
                        utilities[0] = pos_utility
                            
                    if check_in_bound(x, y, next_states[1]): # can go WEST
                        a = actions[1]                              # go north
                        pos_utility = 0
                        # sum (s') { T(s-policy(s)->s') * [R(s) + gamma * value(s')] }
                        reward = grid_mdp.stage_reward(s, a)
                        for next_state in next_states:
                            trans_prob = grid_mdp.get_transition_prob(s, a, next_state)
                            if check_in_bound(x, y, next_state):
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[next_state])
                            else:
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[start])

                        utilities[1] = pos_utility
                    
                    if check_in_bound(x, y, next_states[2]): # can go SOUTH
                        a = actions[2]                              # go north
                        pos_utility = 0
                        # sum (s') { T(s-policy(s)->s') * [R(s) + gamma * value(s')] }
                        reward = grid_mdp.stage_reward(s, a)
                        for next_state in next_states:
                            trans_prob = grid_mdp.get_transition_prob(s, a, next_state)
                            if check_in_bound(x, y, next_state):
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[next_state])
                            else:
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[start])

                        utilities[2] = pos_utility

                    if check_in_bound(x, y, next_states[3]): # can go EAST
                        a = actions[3]                              # go north
                        pos_utility = 0
                        # sum (s') { T(s-policy(s)->s') * [R(s) + gamma * value(s')] }
                        reward = grid_mdp.stage_reward(s, a)
                        for next_state in next_states:
                            trans_prob = grid_mdp.get_transition_prob(s, a, next_state)
                            if check_in_bound(x, y, next_state):
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[next_state])
                            else:
                                pos_utility += trans_prob * (reward + grid_mdp.gamma * value_func[start])

                        utilities[3] = pos_utility

                    if grid_mdp.grid[s] == 0:               # can STAY
                        a = actions [4]
                        # sum over all s' of [trans_prob(s-a->s') * [Reward(s-a->s') + gamma * V_0(s')]]
                        utilities[4] = grid_mdp.get_transition_prob(s, a, s) * (grid_mdp.stage_reward(s, a) + grid_mdp.gamma * value_func[s])

                    max_ind = np.argmax(utilities)
                    policy[s] = actions[max_ind]

                    if policy[s] != past_policy:
                        stable = False

        return value_func, policy
