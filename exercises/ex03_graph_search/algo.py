from abc import ABC, abstractmethod
from dataclasses import dataclass
from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # need to introduce weights!
        pass

@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # BFS but using cost instead of depth when sorting nodes in priority queue

        # todo
        if start == goal:
            Path = [start]
            return Path

        # todo implement here your solution
        Q = [(start, 0)]              # queue

        costToReach = {start: 0}

        # desired outputs:
        parent = {}

        Path = []

        # while the queue still remains
        while Q:
            cur = Q.pop(0)
            
            if cur[0] == goal: # if the goal is found
                cur_node = cur[0]
                while cur_node != start:
                    Path.insert(0, cur_node)
                    cur_node = parent[cur_node]
                Path.insert(0, cur_node)
                return Path

            nodes = self.graph.adj_list[cur[0]]

            for node in nodes:
                c2r = cur[1] + self.graph.get_weight(cur[0], node)

                if node in costToReach.keys():
                    if c2r < costToReach[node]:
                        costToReach[node] = c2r
                        parent[node] = cur[0]
                        Q.append((node, c2r))
                        Q = sorted(Q, key=lambda t: t[1])
                else:
                    costToReach[node] = c2r
                    parent[node] = cur[0]
                    Q.append((node, c2r))
                    Q = sorted(Q, key=lambda t: t[1])

        return Path

@dataclass
class Astar(InformedGraphSearch):

    def heuristic(self, u: X, v: X) -> float:
        # todo
        u_x, u_y = self.graph.get_node_coordinates(u)
        v_x, v_y = self.graph.get_node_coordinates(v)

        # Manhattan Distance:
        dist = ((u_x-v_x)**2 + (u_y-v_y)**2)**0.5
        
        h = dist * TravelSpeed.CITY.value

        return h
        
    def path(self, start: X, goal: X) -> Path:
        # todo
        # todo
        if start == goal:
            Path = [start]
            return Path

        # todo implement here your solution
        Q = [(start, 0)]              # queue

        costToReach = {start: 0}

        # desired outputs:
        parent = {}

        Path = []

        # while the queue still remains
        while Q:            
            cur = Q.pop(0)
            
            if cur[0] == goal: # if the goal is found
                cur_node = cur[0]
                while cur_node != start:
                    Path.insert(0, cur_node)
                    cur_node = parent[cur_node]
                Path.insert(0, cur_node)
                return Path

            nodes = self.graph.adj_list[cur[0]]

            for node in nodes:
                # store cost to reach in costToReach
                # store cost to reach + heuristic in Q
                c2r = costToReach[cur[0]] + self.graph.get_weight(cur[0], node)
                c2r_and_heuristic = c2r + self.heuristic(node, goal)

                if node in costToReach.keys(): # if previously visited...
                    if c2r < costToReach[node]:
                        costToReach[node] = c2r
                        parent[node] = cur[0]
                        Q.append((node, c2r_and_heuristic))
                        Q = sorted(Q, key=lambda t: t[1])
                else:
                    costToReach[node] = c2r
                    parent[node] = cur[0]
                    Q.append((node, c2r_and_heuristic))
                    Q = sorted(Q, key=lambda t: t[1])

        return Path

def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
