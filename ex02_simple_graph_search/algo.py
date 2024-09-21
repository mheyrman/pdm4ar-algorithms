from abc import abstractmethod, ABC
from typing import Tuple

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        # Note: nodes are represented as Integers
        # Note: When a node is expanded, neighbors added to queue in increasing order (small to large int)
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        if start == goal:
            Path = [start]
            OpenedNodes = [start]
            return Path, OpenedNodes

        # todo implement here your solution
        Q = [start]              # queue

        # desired outputs:
        OpenedNodes = []    # opened set
        Visited = [start]
        parent = {}

        Path = []

        # while the queue still remains
        while Q:
            
            cur = Q.pop(0) # FIFO
            OpenedNodes.append(cur)
            
            if cur == goal: # if the goal is found
                while cur != start:
                    Path.insert(0, cur)
                    cur = parent[cur]
                Path.insert(0, start)
                return Path, OpenedNodes

            nodes = graph[cur]
            nodes = sorted(nodes, reverse=True)

            for node in nodes:
                if node not in Visited:
                    Visited.append(node)
                    Q.insert(0, node)
                    parent[node] = cur

        return Path, OpenedNodes


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        if start == goal:
            Path = [start]
            OpenedNodes = [start]
            return Path, OpenedNodes

        # todo implement here your solution
        Q = [start]              # queue

        # desired outputs:
        OpenedNodes = []    # opened set
        Visited = [start]
        parent = {}

        Path = []

        # while the queue still remains
        while Q:
            
            cur = Q.pop(0) # FILO
            OpenedNodes.append(cur)
            
            if cur == goal: # if the goal is found
                while cur != start:
                    Path.insert(0, cur)
                    cur = parent[cur]
                Path.insert(0, start)
                return Path, OpenedNodes

            nodes = graph[cur]
            nodes = sorted(nodes)

            for node in nodes:
                if node not in Visited:
                    Visited.append(node)
                    Q.append(node)
                    parent[node] = cur

        return Path, OpenedNodes


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        if start == goal:
            Path = [start]
            OpenedNodes = [start]
            return Path, OpenedNodes

        # todo implement here your solution
        d = 1               # d <- 1
        Path = []
        OpenedNodes = []    # visited set

        cont = True
        
        while cont:    # while there are still nodes in the queue (max depth has not been reached)
            cont = False

            # desired outputs:
            parent = {}
            depth = {start: 1}

            Q = [start]             # queue
            OpenedNodes = []        # opened set
            Visited = [start]

            # run DFS:
            while Q:
                for node in Q:
                    if depth[node] == d:
                        cont = True
                
                cur = Q.pop(0)      # get first item in stack

                if not depth[cur] > d:      # if the curret node is not above the maximum depth
                    OpenedNodes.append(cur)
                    
                    if cur == goal:     # if the goal is found
                        while cur != start:     # get the path
                            Path.insert(0, cur)
                            cur = parent[cur]
                        Path.insert(0, start)

                        return Path, OpenedNodes

                    nodes = graph[cur]
                    nodes = sorted(nodes, reverse=True)

                    for node in nodes:
                        if node not in Visited and depth[cur] < d:
                            Visited.append(node)
                            Q.insert(0, node)
                            parent[node] = cur
                            depth[node] = depth[cur] + 1
                            
            d = d + 1   # d <- d + 1

        return Path, OpenedNodes
