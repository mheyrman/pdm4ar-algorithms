from abc import ABC, abstractmethod
# from turtle import st
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc


class GridMdp:
    def __init__(self, grid: NDArray[np.int], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        # return the probability of transitioning from state to next_state given an action 
        # P(s-a->s')

        # Note: time = cost that we wanna minimize

        type_cur = self.grid[state]

        (x_cur, y_cur) = state # rows = x, cols = y
        (x_next, y_next) = next_state

        if type_cur == 0:
            if action == 4 and next_state == state:
                return 1.0
            return 0.0
            
        elif type_cur == 1 or type_cur == 2: # if in type Grass
            # check if next state is north, south, west, or east
            if (y_cur == y_next) and (x_cur - 1 == x_next): # next state is north
                if action == 0: # go north
                    return 0.75
                return 0.25/3.0

            elif (y_cur-1 == y_next) and (x_cur == x_next):
                if action == 1: # go west
                    return 0.75
                return 0.25/3.0
            
            elif (y_cur == y_next) and (x_cur + 1 == x_next):
                if action == 2:
                    return 0.75
                return 0.25/3.0
            
            elif (y_cur + 1 == y_next) and (x_cur == x_next):
                if action == 3:
                    return 0.75
                return 0.25/3.0
            
            # next state is stay
            return 0.0
        
        elif type_cur == 3: # if in type Swamp
            if action == 4:
                return 0.0
            elif (y_cur == y_next) and (x_cur - 1 == x_next): # next state is north
                if action == 0: # go north
                    return 0.5
                return 0.25/3.0
            
            elif (y_cur-1 == y_next) and (x_cur == x_next):
                if action == 1: # go west
                    return 0.5
                return 0.25/3.0

            elif (y_cur == y_next) and (x_cur + 1 == x_next):
                if action == 2:
                    return 0.5
                return 0.25/3.0
            
            elif (y_cur + 1 == y_next) and (x_cur == x_next):
                if action == 3:
                    return 0.5
                return 0.25/3.0
            return 0.25

    def stage_reward(self, state: State, action: Action) -> float:
        # todo
        cell_type = self.grid[state]

        if cell_type == 0:
            return 10.0
        elif cell_type == 1 or cell_type == 2:
            return -1.0
        else:
            return -2.0

class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
