import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

# my imports:
from dg_commons import SE2Transform
import shapely
from shapely.errors import ShapelyDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import time
import math

from pdm4ar.exercises.ex05.structures import *


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    prm_graph: dict[set] = None
    prm_points: list = None

    def __init__(self,
                 sg: VehicleGeometry,
                 sp: VehicleParameters
                 ):
        self.sg = sg
        self.sp = sp
        self.name: PlayerName = None
        self.goal: PlanningGoal = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

        # my stuff
        self.full_path: List = []
        # self.x_points: list = []
        # self.y_points: list = []
        # self.prm_points: list = []
        self.dynamic_obstacles = []
        self.dynamic_obs_count = 0.0
        self.path_edges = []
        self.path_points: List = []
        self.path_line: shapely.geometry.LineString() = None
        self.old_line = None
        # radius = DubinsParam(wheel_base / np.tan(max_steering_angle))
        self.turning_radius = ((sg.lr + sg.lf) / np.tan(sp.delta_max))
        self.current_line = None
        self.generate_path: bool = True
        self.bounds: shapely.geometry.Polygon = None
        self.prev_delta_car = 0.0
        self.prev_delta_des = 0.0
        self.speed = 0.0
        self.recalculated = False
        self.seen_vehicles = []
        self.go_go = True
        self.union = None


        self.angle_dict = {}


    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles.values())

        # get map bounds
        lanelet_list = []
        for lanelet in self.lanelet_network.lanelet_polygons:
            lanelet_list.append(lanelet.shapely_object)
        lanelet_list.append(self.goal.goal)
        self.union = shapely.ops.unary_union(lanelet_list)
        

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # todo implement here some better planning
        
        # from question @489 on Piazza:
        # 1. A module which computes a path from the start to the goal which is collision free 
        #   (although not necessarily kinodynamically feasible for our car).
        # 2. A module which takes the path of the previous module and makes it dynamically feasible
        # 3. For the controller you could start off by implementing a simple controller which tracks 
        #   a static reference and then incrementally increase the complexity of the controller (if 
        #   you google kinematic bycicle model controller you can find a bunch of different algorithms)

        # NOTE:
        # Calculate RRT path
        # every iteration, check if there are collisions
        # if there are collisions, calculate new RRT path until no collisios
        # continue

        # cur state options: x, y, psi, vx, delta, occupancy(polygon)

        cur_state = sim_obs.players[self.name].state
        start_pos = shapely.geometry.Point(cur_state.x, cur_state.y)

        # find goal point
        goal_min_x, goal_min_y, goal_max_x, goal_max_y = self.goal.goal.exterior.bounds
        goal_pos_uncentered = ((goal_max_x + goal_min_x) / 2.0, (goal_max_y + goal_min_y) / 2.0)

        # project path points to nearest lane center
        goal_min_dist = np.inf
        goal_nearest_lane = None

        for lanelet in self.lanelet_network.lanelets:
            center_line = shapely.geometry.LineString(lanelet.center_vertices)
            dist_cur = shapely.geometry.Point(goal_pos_uncentered[0], goal_pos_uncentered[1]).distance(center_line)
            
            if dist_cur < goal_min_dist:
                goal_min_dist = dist_cur
                goal_nearest_lane = center_line

        goal_pos_point = goal_nearest_lane.interpolate(goal_nearest_lane.project(shapely.geometry.Point(goal_pos_uncentered[0], goal_pos_uncentered[1])))
        goal_pos = (goal_pos_point.x, goal_pos_point.y)

        # calculate the car's actual bounds:
        b1 = shapely.geometry.Point(cur_state.x + self.sg.lf * np.cos(cur_state.psi) + 2.0 * self.sg.w_half * np.cos(cur_state.psi + np.pi / 2.0),
                                    cur_state.y + self.sg.lf * np.sin(cur_state.psi) + 2.0 * self.sg.w_half * np.sin(cur_state.psi + np.pi / 2.0))
        b2 = shapely.geometry.Point(cur_state.x + self.sg.lf * np.cos(cur_state.psi) - 2.0 * self.sg.w_half * np.cos(cur_state.psi + np.pi / 2.0),
                                    cur_state.y - self.sg.lf * np.sin(cur_state.psi) + 2.0 * self.sg.w_half * np.sin(cur_state.psi + np.pi / 2.0))
        b3 = shapely.geometry.Point(cur_state.x - self.sg.lf * np.cos(cur_state.psi) - 2.0 * self.sg.w_half * np.cos(cur_state.psi + np.pi / 2.0),
                                    cur_state.y - self.sg.lf * np.sin(cur_state.psi) - 2.0 * self.sg.w_half * np.sin(cur_state.psi + np.pi / 2.0))
        b4 = shapely.geometry.Point(cur_state.x - self.sg.lf * np.cos(cur_state.psi) + 2.0 * self.sg.w_half * np.cos(cur_state.psi + np.pi / 2.0),
                                    cur_state.y + self.sg.lf * np.sin(cur_state.psi) - 2.0 * self.sg.w_half * np.sin(cur_state.psi + np.pi / 2.0))

        self.bounds = shapely.geometry.Polygon([[b1.x, b1.y], [b2.x, b2.y], [b3.x, b3.y] ,[b4.x, b4.y]])

        while self.generate_path or self.dynamic_obs_count > 0.0:
            print("=================================================")
            print(self.name)
            t = time.time()
            self.generate_path = False

            self.path_edges = []
            self.path_points = []

            # rrt_edges = self.calculate_path(sim_obs, start_pos) # -> List(tuple(Point, Point))
            rrt_star_edges  = self.calculate_rrt_star(sim_obs, start_pos, goal_pos)

            if rrt_star_edges == []:
                self.go_go = False
                break
            # self.path_edges = rrt_star_edges
            # extract only the final path as list of tuples
            cur = list(rrt_star_edges)[-1]
            pe = []
            pp = []

            # Adding RRT* calcualted line to show in visualization
            while cur in rrt_star_edges:
                prev = rrt_star_edges[cur]
                pe.insert(0, (prev, cur))
                pp.insert(0, cur)
                cur = prev
            pp.insert(0, (start_pos.x, start_pos.y))
            self.old_line = shapely.geometry.LineString(pp)
            # self.customViz1(sim_obs)

            # smoothen path by finding shortest possible connections for all nodes starting from goal
            rrt_star_edges = self.optimize_path(sim_obs, start_pos, goal_pos, rrt_star_edges)

            cur = list(rrt_star_edges)[-1]
            while cur in rrt_star_edges:
                if cur in self.angle_dict:
                    print(self.angle_dict[cur])
                prev = rrt_star_edges[cur]

                # prev_point = shapely.geometry.Point(prev[0], prev[1])
                # cur_point = shapely.geometry.Point(cur[0], cur[1])

                self.path_edges.insert(0, (prev, cur))
                self.path_points.insert(0, cur)
                cur = prev
            self.path_points.insert(0, (start_pos.x, start_pos.y))
            self.path_line = shapely.geometry.LineString(self.path_points)
            
            self.current_line = self.path_edges.pop(0)
            self.full_path.append(self.current_line)

            if self.dynamic_obs_count > 0.0:
                self.dynamic_obs_count = 0.0
                self.dynamic_obstacles = []

            print(time.time() - t)
            print("=================================================")

        if not self.go_go:
            v_des = 0.0
            k_d = 5.0

            a = k_d * (v_des - cur_state.vx)

            if a < -8.0:
                a = -8.0
            elif a > 8.0:
                a = 8.0

            return VehicleCommands(acc=a, ddelta=0.0)

        k_d = 0.02
        # v_des = 10.0
        # k_p_delta = 0.55
        # k_d_delta = 0.2
        # k_dd = 0.4
        # v_des = 15.0 # idk man
        # k_p_delta = 0.6
        # k_d_delta = 0.3
        # k_dd = 0.31
        # v_des = 25.0 # idk man
        # k_p_delta = 0.7
        # k_d_delta = 0.45
        # k_dd = 0.2
        v_des = 15.0
        true_v_des = v_des
        k_p_delta = 0.75
        k_d_delta = 0.33
        k_dd = 0.4
        # v_des = 40.0 # idk man
        # k_p_delta = 0.95
        # k_d_delta = 0.6
        # k_dd = 0.13

        # check dynamic objects
        # https://github.com/ivanalberico/Planning-and-Decision-Making-for-Autonomous-Robots-ETH/blob/899998849bddd3460753b3df4213c7aa72d87814/src/pdm4ar/exercises/final21/agent.py#L103
        
        # need to think of separate possible states:
        # car in front and coming straight towards path within 2 seconds -> car w/ smaller number stop, other car replan around
        # car coming from right to intersects path : stop
        # car is coming from the left/front : continue unless it's on the path, then path plan around it

        # find if car is facing towards or away from car -> difference between headings
        # whether car is gonna crash -> heading * velocity for 2 seconds (40 time steps)

        # to prevent one meeting from causing hell -> keep list of cars used to recalculate every time and only recalculate once per car
        # 

        car_keys = list(sim_obs.players.keys())[1:] # vroom vroom
        for car in car_keys:
            print(str(self.name) + " sees " + str(car))
            print(self.name < car)
            if sim_obs.players[car].occupancy.intersects(shapely.geometry.Point(cur_state.x, cur_state.y).buffer(7.0)):
                print("oh no")
                seen_car = sim_obs.players[car]
                seen_state = seen_car.state
                # v_des = 0.0
                # k_d = 5.0
                # print(seen_car.occupancy.coords)
                # if car is to the right:
                # using v, calculate pose over next 2 seconds (40 time steps)
                # if that line is in front of our car and intersects our path, stop
                # otherwise, assume it's fine

                gamma = np.arctan2(seen_state.y - cur_state.y, seen_state.x - cur_state.x) - cur_state.psi
                gamma = math.remainder(gamma, math.tau)

                predicted_poses = [shapely.geometry.Point(seen_state.x, seen_state.y)]
                vel = (seen_state.vx * np.cos(seen_state.psi), seen_state.vx * np.sin(seen_state.psi))
                if seen_state.vx < 0.1:
                    if shapely.geometry.Point(seen_state.x, seen_state.y).buffer(2.5).intersects(self.path_line):
                        v_des = 0.0
                        k_d = 5.0
                        break
                    vel = (true_v_des * np.cos(seen_state.psi), true_v_des * np.sin(seen_state.psi))

                next_x = predicted_poses[-1].x + vel[0] * 2.0
                next_y = predicted_poses[-1].y + vel[1] * 2.0
                predicted_poses.append(shapely.geometry.Point(next_x, next_y))

                predicted_trajectory = shapely.geometry.LineString(predicted_poses)
                # if the car is predicted to intersect with our path
                if predicted_trajectory.buffer(2.1 * self.sg.w_half).intersects(self.path_line):
                    path_intersection = predicted_trajectory.buffer(2.1 * self.sg.w_half).intersection(self.path_line)
                    if path_intersection.geom_type == 'LineString':
                        points = path_intersection.coords
                        min_point = None
                        min_dist = np.inf
                        for p in points:
                            dist = np.sqrt((p[1] - seen_state.y) ** 2 + (p[0] - seen_state.x) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                min_point = shapely.geometry.Point(p[0], p[1])
                        path_intersection = min_point
                        gamma = np.arctan2(path_intersection.y - cur_state.y, path_intersection.x - cur_state.x) - cur_state.psi
                        gamma = math.remainder(gamma, math.tau)
                        print("gamma: " + str(gamma))
                        if np.abs(gamma) < np.pi/2.0:
                            v_des = 0.0
                            k_d = 5.0

                    #     if gamma > 0 and gamma < np.pi: # right of way to right cars
                    #         v_des = 0.0
                    #     elif np.abs(gamma) < np.pi/4.0: # if the car is in front -> recalculate
                    #         # if self.name < car:
                    #         if not self.recalculated:
                    #             self.generate_path = True
                    #             self.dynamic_obstacles.append(seen_car.occupancy)
                    #             self.dynamic_obs_count += 1.0
                    #             self.seen_vehicles.append(car)
            elif sim_obs.players[car].occupancy.intersects(shapely.geometry.Point(cur_state.x, cur_state.y).buffer(35.0)):
                seen_car = sim_obs.players[car]
                seen_state = seen_car.state

                gamma = np.arctan2(seen_state.y - cur_state.y, seen_state.x - cur_state.x) - cur_state.psi
                gamma = np.abs(math.remainder(gamma, math.tau))

                if seen_state.vx < 0.1 and seen_car.occupancy.intersects(self.path_line.buffer(1.5 * self.sg.w_half)) and gamma < np.pi / 2.0:
                    self.dynamic_obstacles.append(seen_car.occupancy)
                    self.dynamic_obs_count += 1
                    self.seen_vehicles.append(car)

                            
        # controls
        car_front_pos = shapely.geometry.Point(cur_state.x + np.cos(cur_state.psi) * self.sg.lf, cur_state.y + np.sin(cur_state.psi) * self.sg.lf)
        car_rear_pos = shapely.geometry.Point(cur_state.x - np.cos(cur_state.psi) * self.sg.lr, cur_state.y - np.sin(cur_state.psi) * self.sg.lr)
        cur_vel = cur_state.vx

        if self.speed == 0.0:
            self.speed = 1.0 # set -1 eventually...
            if random.random() < 0.5:
                self.speed = 1.0
            # print(self.speed)
        
        # Pure Pursuit:
        ld = k_dd * true_v_des
        angles = np.linspace(cur_state.psi-np.pi/2.5, cur_state.psi+np.pi/2.5, 5)
        
        circle_x = car_rear_pos.x + np.cos(angles) * ld
        circle_y = car_rear_pos.y + np.sin(angles) * ld
        circle = shapely.geometry.LineString(zip(circle_x, circle_y))

        if circle.intersects(self.goal.goal) or self.bounds.intersects(self.goal.goal):
            alpha = 0.0
        else:
            if self.path_line.intersects(circle):
                # multiple intersections cause an attribute error 
                # when that happens, take the point most 'in front' of the car
                ld_point = self.path_line.intersection(circle)
                if ld_point.geom_type == 'MultiPoint':
                    points = [(p.x, p.y) for p in ld_point]
                    min_point = None
                    min_gamma = np.inf
                    for p in points:
                        gamma = math.remainder(np.arctan2(p[1] - car_rear_pos.y, p[0] - car_rear_pos.x) - cur_state.psi, math.tau)
                        if gamma < min_gamma:
                            min_gamma = gamma
                            min_point = shapely.geometry.Point(p[0], p[1])

                    ld_point = min_point
                    # ld_point = shapely.geometry.Point(ld_point[0].x, ld_point[0].y)
                    # self.generate_path = True
                alpha = np.arctan2(ld_point.y - car_rear_pos.y, ld_point.x - car_rear_pos.x) - cur_state.psi
            elif v_des > 0.0:
                # self.generate_path = True
                # if lost just stop -> not worth potential infinite loop
                v_des = 0.0
                k_d = 5.0
                alpha = 0.0


        alpha = math.remainder(alpha, math.tau)

        delta = np.arctan2(2.0 * (self.sg.lf + self.sg.lr) * np.sin(alpha), ld)

        ddelta = k_p_delta * (delta - cur_state.delta) + k_d_delta * ((delta - self.prev_delta_des) / 0.05 - (cur_state.delta - self.prev_delta_car) / 0.05)

        if ddelta < -0.5:
            ddelta = -0.5
        elif ddelta > 0.5:
            ddelta = 0.5

        a = k_d * (v_des - cur_vel)

        if a < -8.0:
            a = -8.0
        elif a > 8.0:
            a = 8.0

        # print(a)
        
        self.customViz(sim_obs, circle_x, circle_y)

        # if not shapely.geometry.LineString(self.current_line).intersects(self.bounds):
        #     self.generate_path = True

        if self.bounds.contains(shapely.geometry.Point(self.current_line[1])):
            self.current_line = self.path_edges.pop(0)
            self.full_path.append(self.current_line)

        return VehicleCommands(acc=a, ddelta=ddelta)

    def optimize_path(self, sim_obs, start, goal_pos, rrt_star_edges):
        # cur = goal_pos # start at goal

        # # if goal length > 10 -> divide into parts of length 5.0
        # if cur in rrt_star_edges:
        #     child = rrt_star_edges[cur]
        #     l = np.sqrt((cur[1] - child[1]) ** 2 + (cur[0] - child[0]) ** 2)

        #     if l >= 10:
        #         theta = np.arctan2(child[1] - cur[1], child[0] - cur[0])
        #         # divide:
        #         while l >= 5.0:
        #             new_child = (cur[0] + 5.0 * np.cos(theta), cur[1] + 5.0 * np.sin(theta))
        #             rrt_star_edges[cur] = new_child
        #             cur = new_child
        #             l -= 5.0
        #         rrt_star_edges[cur] = child
        cur = goal_pos
        start = (start.x, start.y)

        print("rewiring")
        t = time.time()

        theta_threshold = np.pi / 5.0
        child = None

        while cur in rrt_star_edges:
            parent = rrt_star_edges[cur]
            changed = True

            # check every single ancestor
            while changed:
                if parent in rrt_star_edges: # if we haven't reached start

                    grandparent = rrt_star_edges[parent]
                    angle = self.check_angles(rrt_star_edges, start, grandparent, cur, sim_obs.players[self.name].state)

                    if angle < theta_threshold:
                        if cur != goal_pos:
                            temp_dict = {cur: grandparent}
                            forward_angle = self.check_angles(temp_dict, start, cur, child, sim_obs.players[self.name].state)
                            if forward_angle < theta_threshold:
                                safe = self.check_safety_certificate(grandparent, cur)
                            else:
                                safe = False
                        else:
                            safe = self.check_safety_certificate(grandparent, cur)
                    else:
                        safe = False

                    if safe:
                        rrt_star_edges[cur] = grandparent
                        parent = rrt_star_edges[cur]
                    else:
                        parent = grandparent

                else: # done looping
                    changed = False
                    child = cur
                    cur = rrt_star_edges[cur]
        
        print(time.time() - t)
        
        return rrt_star_edges

    # p1 = previous
    # p2 = middle
    # p3 = next
    # get angle going previous -> middle -> next
    def check_angles(self, edges, start, p2, p3, cur_state) -> float:
        if start != p2:
            p1 = edges[p2]
        else:                           # only contains start node: calculate difference between heading and line
            heading = cur_state.psi
            p1 = (p2[0] - np.cos(heading), p2[1] - np.sin(heading))

        # v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        # v2 = np.array([p1[0] - p3[0], p1[1] - p3[1]])

        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        psi = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        if psi > np.pi or psi < -np.pi:
            psi = 2.0 * np.pi - psi
        elif psi < -np.pi:
            psi = 2.0 * np.pi + psi
        return np.abs(math.remainder(psi, math.tau))

    def calculate_rrt_star(self, sim_obs: SimObservations, start_pos: shapely.geometry.Point, goal_pos):
        self.angle_dict = {}

        start = (start_pos.x, start_pos.y)

        t = time.time()

        # bounds of lanes
        x_min, y_min, x_max, y_max = self.union.bounds

        # # implement RRT* from here:
        # r = 8.0
        # if random.random() > 0.5:
        r = 5.0
        r = 4.0
        r = 3.0
        theta_threshold = np.pi / 5.2 # use 4.0 if slow
        # if random.random() > 0.5:
        #     r = 5.0

        while True:
            if self.dynamic_obs_count > 0.0:
                if (time.time() - t) > 15.0:
                    return []
            elif (time.time() - t) > 40.0:
                return []
            print("checking new path")
            edges = {} # next: prev
            vertices = {start: 0.0}
            stuck = False

            for i in range(0, 350):
                new_point = start

                for j in range(0,300):
                    if j == 300 - 1:
                        stuck = True

                    # point selection:
                    if random.random() < 0.98:
                        while True:
                            sample = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
                            if self.union.contains(shapely.geometry.Point(sample[0], sample[1])): # .buffer(1.65 * self.sg.w_half)
                                break
                        nearest = self.get_nearest(vertices.keys(), sample)   # nearest point to new point!
                    else:
                        sample = goal_pos

                        nearest = self.get_nearest(vertices.keys(), sample)   # nearest point to new point!
                        goal_ang = self.check_angles(edges,start, nearest, sample, sim_obs.players[self.name].state)
                        if goal_ang < theta_threshold:
                            goal_safe = self.check_safety_certificate(nearest, sample)
                        else:
                            goal_safe = False
                        # if path to goal is clean -> just go there
                        if goal_safe:# and goal_ang < theta_threshold:
                            new_point = sample
                            self.angle_dict[new_point] = angle
                            edges[new_point] = nearest

                            cost = vertices[nearest] + np.sqrt((nearest[1] - new_point[1]) ** 2 + (nearest[0] - new_point[0]) ** 2)
                            vertices[new_point] = cost
                            break
                
                    new_point = self.steer(nearest, sample, r)          # new point that should also be on a lane!

                    # check if the angle is acceptable
                    angle = self.check_angles(edges, start, nearest, new_point, sim_obs.players[self.name].state)
                    if angle < theta_threshold:
                        safe = self.check_safety_certificate(nearest, new_point)
                    else:
                        safe = False
                    if safe:# and angle < theta_threshold:   # point has been successfully chosen!
                        self.angle_dict[new_point] = angle
                        edges[new_point] = nearest

                        cost = vertices[nearest] + np.sqrt((nearest[1] - new_point[1]) ** 2 + (nearest[0] - new_point[0]) ** 2)
                        vertices[new_point] = cost
                        break
                
                if stuck:
                    break
                
                neighbors = self.get_neighbors(vertices.keys(), new_point, 1.0 * r)
                for n in neighbors:
                    # try:
                    n_cost = vertices[new_point] + np.sqrt((n[1] - new_point[1]) ** 2 + (n[0] - new_point[0]) ** 2)
                    # except Exception as e:
                    #     print("weirdness")
                    #     pass
                    if  n_cost < vertices[n]:
                        # check safety between neighbor and new point:
                        angle = self.check_angles(edges, start, new_point, n, sim_obs.players[self.name].state)
                        if angle < theta_threshold: 
                            safe = self.check_safety_certificate(new_point, n)
                        else:
                            safe = False
                        if safe:# and angle < theta_threshold:   # point has been successfully chosen!
                            self.angle_dict[n] = angle
                            edges[n] = new_point
                            vertices[n] = n_cost

                if self.goal.goal.contains(shapely.geometry.Point(new_point[0], new_point[1])):
                    print("i: " + str(i))
                    return edges
                elif self.check_path_to_goal(edges, start, new_point, goal_pos, sim_obs, theta_threshold): # if a path to goal from the last edge is possible -> just go there!
                    edges[goal_pos] = new_point
                    self.angle_dict[goal_pos] = 0.0

                    cost = vertices[new_point] + np.sqrt((new_point[1] - goal_pos[1]) ** 2 + (new_point[0] - goal_pos[0]) ** 2)
                    vertices[goal_pos] = cost
                    print("i: " + str(i))
                    return edges

            
    def check_path_to_goal(self, edges, start, last_pos, goal_pos, sim_obs, theta_threshold) -> bool:
        goal_ang = self.check_angles(edges, start, last_pos, goal_pos, sim_obs.players[self.name].state)
        if goal_ang < theta_threshold:
            goal_safe = self.check_safety_certificate(last_pos, goal_pos)
        else:
            goal_safe = False
        # if path to goal is clean -> just go there
        if goal_safe:# and goal_ang < theta_threshold:
            return True
        return False

    def get_neighbors(self, vertices, new_point, radius):
        neighbors = []
        for v in vertices:
            if np.sqrt((v[1] - new_point[1]) ** 2 + (v[0] - new_point[0]) ** 2) <= radius:
                neighbors.append(v)

        return neighbors              
                
    def check_safety_certificate(self, p1, p2) -> bool:
        r = 1.7 * self.sg.w_half  # 1.65 is nice

        for dyn in self.dynamic_obstacles:
            if dyn.intersects(shapely.geometry.LineString([(p1[0], p1[1]), (p2[0], p2[1])]).buffer(r)):
                return False

        for obs in self.static_obstacles:
            if obs.shape.intersects(shapely.geometry.LineString([(p1[0], p1[1]), (p2[0], p2[1])]).buffer(r)):
                return False
        if not self.union.contains(shapely.geometry.LineString([(p1[0], p1[1]), (p2[0], p2[1])]).buffer(r)):
            # the union should contain the entirety of the line + some buffer
            return False

        return True


    def steer(self, p1: Tuple, p2: Tuple, r: float) -> Tuple:
        new_point = ()

        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        uncentered = (p1[0] + r * np.cos(theta), p1[1] + r * np.sin(theta))

        return uncentered

        # project path points to nearest lane center
        # lanelets have center lines defined by 2 points: lanelet._
        min_dist = np.inf
        nearest_lanelet_center= None

        for lanelet in self.lanelet_network.lanelets:
            center_line = shapely.geometry.LineString(lanelet.center_vertices)
            dist_cur = shapely.geometry.Point(uncentered[0], uncentered[1]).distance(center_line)
            
            if dist_cur < min_dist:
                min_dist = dist_cur
                nearest_lanelet_center = center_line

        new_point = nearest_lanelet_center.interpolate(nearest_lanelet_center.project(shapely.geometry.Point(uncentered[0], uncentered[1])))

        return (new_point.x, new_point.y)
            
    def get_nearest(self, vertices: List[Tuple], point: Tuple) -> Tuple:
        while True:
            nearest = None
            min_dist = np.inf
            # find closest point
            for point_candidate in vertices:
                dist = np.sqrt((point[0] - point_candidate[0]) ** 2 + (point[1] - point_candidate[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = point_candidate

            return nearest
              
    def point_in_obstacle(self, point: shapely.geometry.Point) -> bool:
        for obs in self.static_obstacles:
            if obs.shape.contains(point):
                return True
        return False

#######################################################################################################
#######################################################################################################
#######################################################################################################

    def customViz(self, sim_obs, circle_x, circle_y):
        plt.figure()
        for i in range(len(self.static_obstacles)):
            if type(self.static_obstacles[i].shape) == shapely.geometry.LinearRing or \
                type(self.static_obstacles[i].shape) == shapely.geometry.LineString:
                plt.fill(*self.static_obstacles[i].shape.xy, facecolor='grey')
            else:
                plt.fill(*self.static_obstacles[i].shape.exterior.xy, facecolor='red')
        
        plt.scatter(sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y, color='orange')
        plt.plot(*self.old_line.xy, 'green')
        plt.plot(*self.path_line.xy) # Equivalent
        plt.plot(circle_x, circle_y, 'm')
        
        for start, end in self.full_path:
            #print(np.sqrt((end.y-start.y) ** 2 + (end.x-start.x) ** 2))
            plt.plot([start[0], end[0]],[start[1], end[1]], 'k')

        if self.current_line:
            plt.plot([self.current_line[0][0], self.current_line[1][0]], [self.current_line[0][1], self.current_line[1][1]], 'r')
        
            
        plt.fill(*self.goal.goal.exterior.xy, facecolor='green')
        
        plt.gca().set_aspect('equal')
        plt.savefig(str(self.name)+'.png')
        plt.close()
        return