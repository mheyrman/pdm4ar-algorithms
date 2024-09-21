import numpy as np
import shapely

from typing import List
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
        Triangle: CollisionPrimitives.circle_triangle_collision,
        Polygon: CollisionPrimitives.circle_polygon_collision,
        Circle: CollisionPrimitives.circle_circle_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert (
        type(p_2) in COLLISION_PRIMITIVES[type(p_1)]
    ), "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # Strategy:
        #   1. check if the path segmets collide with any obstacles!
        #   1a. SPECIAL CIRCLE CASE: double circle obstacle radius and check
        #   2. draw circles at every obstacle vertex of radius of robot size
        #   3. check if they collide with the path
        #   4. draw circles at every path vertex of radius of robot size
        #   5. check if they collife with the obstacles (this might suck)
        collided_segments = []

        waypoints = t.waypoints

        path_segments = []
        path_circles = []
        if len(waypoints) > 1:
            for i in range(0, len(waypoints) - 1):
                path_segments.append(Segment(waypoints[i], waypoints[i+1]))
                path_circles.append(Circle(waypoints[i], r))
        path_circles.append(Circle(waypoints[-1], r))

        l = len(path_segments) # REPLACE L INTO FOR STATEMENT LATER: VS INSISTS ON USING ENUMERATE
        for i in range(l):   # MAY BE ABLE TO COMBINE THIS AND ABOVE LOOP DEPENDING ON 
                                    # IF PATH_SEGMENTS IS NEEDED AGAIN IN ANOTHER LOOP
            seg = path_segments[i]
            
            for obs in obstacles:
                if isinstance(obs, Circle):
                    # special circle case: EXPAND RADIUS
                    new_circle = Circle(obs.center, obs.radius + r)
                    if CollisionPrimitives.circle_segment_collision(new_circle, seg):
                        collided_segments.append(i)
                        break # since we already check if the robot's radius crashes we always break w/ a circle
                    continue     
                elif isinstance(obs, Triangle):
                    if CollisionPrimitives.triangle_segment_collision(obs, seg):
                        collided_segments.append(i)
                        break
                
                    vertices = [obs.v1, obs.v2, obs.v3]
                    
                    for v in vertices:
                        corner_circle = Circle(v, r)

                        if CollisionPrimitives.circle_segment_collision(corner_circle, seg):
                            collided_segments.append(i)
                            break

                    triangle_segs = [Segment(vertices[0], vertices[1]),
                                    Segment(vertices[0], vertices[2]),
                                    Segment(vertices[1], vertices[2])]
                    
                    for t_seg in triangle_segs:
                        if (CollisionPrimitives.circle_segment_collision(path_circles[i], t_seg) or
                            CollisionPrimitives.circle_segment_collision(path_circles[i + 1], t_seg)):
                            collided_segments.append(i)
                            break
                    continue
                elif isinstance(obs, Polygon):
                    if CollisionPrimitives.polygon_segment_collision_aabb(obs, seg):
                        collided_segments.append(i)
                        break
                    
                    p_segs = []
                    
                    for j in range(0, len(obs.vertices) - 1):
                        v = obs.vertices[j]
                        p_segs.append(Segment(v, obs.vertices[j+1]))
                        corner_circle = Circle(v, r)

                        if CollisionPrimitives.circle_segment_collision(corner_circle, seg):
                            collided_segments.append(i)
                            break
                    corner_circle = Circle(obs.vertices[-1], r)
                    if CollisionPrimitives.circle_segment_collision(corner_circle, seg):
                        collided_segments.append(i)
                        break
                    p_segs.append(Segment(obs.vertices[-1], obs.vertices[0]))
                    
                    for p_seg in p_segs:
                        if (CollisionPrimitives.circle_segment_collision(path_circles[i], p_seg) or
                            CollisionPrimitives.circle_segment_collision(path_circles[i + 1], p_seg)):
                            collided_segments.append(i)
                            break
                    continue

        return collided_segments

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        
        # Strategy:
        #   1. Create occupancy binary occupancy grid
        #   2. Fill with the occupancy of obstacles
        #   3. Determine where the path passes and if there's an intersection with
        #       already occupied space, add that segment to the list of intersections!

        collided_segments = []

        p_min, p_max = t.get_boundaries()
        
        # grid size of 1
        # use 2.0 * r to create enough buffer otherwise it misses things!
        grid_size = (np.round((p_max.x - p_min.x) + 2.0 * r).astype(int) + 1, np.round((p_max.y - p_min.y) + 2.0 * r).astype(int) + 1)
        occ_grid = np.zeros(grid_size)
        occupied_grids = []

        # first check every obstacle
        for obs in obstacles:
            obs_min, obs_max = obs.get_boundaries()
            bottom_x = np.floor(obs_min.x - r).astype(int)
            top_x = np.floor(obs_max.x + r).astype(int)
            bottom_y = np.floor(obs_min.y - r).astype(int)
            top_y = np.floor(obs_max.y + r).astype(int)

            for x in range(bottom_x, top_x):
                for y in range(bottom_y, top_y):
                    if x >= -(r + p_min.x) and x < (grid_size[0] - r) and y >= -(r + p_min.y) and y < (grid_size[1] - r):
                        p = Point(x, y)
                        # since grid size is 1, use circle of r = 0.5
                        # for speed -> implemented check_collision to check all obstacles with a circle!
                        if check_collision(Circle(p, np.sqrt(1)), obs):
                            occ_grid[np.round(x - p_min.x + r).astype(int), np.round(y - p_min.y + r).astype(int)] = 1
                            occupied_grids.append(Point(x, y))

        waypoints = t.waypoints
        for i in range(0, len(waypoints) - 1):
            seg = Segment(waypoints[i], waypoints[i+1])
            for p in occupied_grids:
                # 1.5 * r = 0.98
                c = Circle(p, r)
                if CollisionPrimitives.circle_segment_collision(c, seg):
                    collided_segments.append(i)
                    break

        return collided_segments

    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        collided_segments = []

        polygons = []
        for obs in obstacles:
            p_min, p_max = obs.get_boundaries()
            vertices = [(p_min.x, p_min.y), (p_min.x, p_max.y), (p_max.x, p_max.y), (p_max.x, p_min.y)]
            polygons.append(shapely.geometry.Polygon(vertices))

        tree = shapely.strtree.STRtree(polygons)

        waypoints = t.waypoints

        for i in range(0, len(waypoints) - 1):
            p1 = shapely.geometry.Point(waypoints[i].x, waypoints[i].y)
            p2 = shapely.geometry.Point(waypoints[i+1].x, waypoints[i+1].y)
            seg = Segment(waypoints[i], waypoints[i+1])
            ind = tree.nearest(p1)
            ind2 = tree.nearest(p2)
            escape = False

            for j, poly in enumerate(polygons):
                if poly == ind or poly == ind2:
                    obs = obstacles[j]
                    if (isinstance(obs, Circle)):
                        if CollisionPrimitives.circle_segment_collision(Circle(obs.center, obs.radius + r), seg):
                            collided_segments.append(i)
                            break
                    else:
                        if isinstance(obs, Triangle):
                            obs = Polygon([obs.v1, obs.v2, obs.v3])
                        
                        if CollisionPrimitives.polygon_segment_collision_aabb(obs, seg):
                            collided_segments.append(i)
                            break

                        segments = []
                        num = len(obs.vertices)
                        for n in range(num - 1):
                            c = Circle(obs.vertices[n], r)
                            if CollisionPrimitives.circle_segment_collision(c, seg):
                                collided_segments.append(i)
                                escape = True
                                break
                            segments.append(Segment(obs.vertices[n], obs.vertices[n+1]))
                        if escape:
                            break
                        c = Circle(obs.vertices[-1], r)
                        if CollisionPrimitives.circle_segment_collision(c, seg):
                            collided_segments.append(i)
                            escape = True
                            break
                        segments.append(Segment(obs.vertices[-1], obs.vertices[0]))

                        c1 = Circle(waypoints[i], r)
                        c2 = Circle(waypoints[i + 1], r)

                        for s in segments:
                            if (CollisionPrimitives.circle_segment_collision(c1, s) or
                                CollisionPrimitives.circle_segment_collision(c2, s)):
                                collided_segments.append(i)
                                escape = True
                                break
                        if escape:
                            break
        return collided_segments
            
    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        p1_tmp = Point(current_pose.p[0], current_pose.p[1])
        p2_tmp = Point(next_pose.p[0], next_pose.p[1])

        dist_cc = np.sqrt((p2_tmp.y - p1_tmp.y) ** 2 + (p2_tmp.x - p1_tmp.x) ** 2)

        p1 = Point(0, 0)
        p2 = Point(dist_cc, 0)

        seg = Segment(p1, p2)
        path_circles = [Circle(p1, r), Circle(p2, r)]

        for obs in observed_obstacles:
            if isinstance(obs, Circle) and CollisionPrimitives.circle_segment_collision(Circle(obs.center, obs.radius + r), seg):
                return True
            elif isinstance(obs, Triangle):
                vertices = [obs.v1, obs.v2, obs.v3]
                segments = [Segment(obs.v1, obs.v2), Segment(obs.v2, obs.v3), Segment(obs.v3, obs.v1)]
                if CollisionPrimitives.triangle_segment_collision(obs, seg):
                    return True
                for v in vertices:
                    if CollisionPrimitives.circle_segment_collision(Circle(v, r), seg):
                        return True
                for s in segments:
                    if (CollisionPrimitives.circle_segment_collision(path_circles[0], s) or
                        CollisionPrimitives.circle_segment_collision(path_circles[1], s)):
                        return True
            elif isinstance(obs, Polygon):
                vertices = obs.vertices
                segments = []
                num = len(vertices)
                if CollisionPrimitives.polygon_segment_collision_aabb(obs, seg):
                    return True
                for i in range(num - 1):
                    v = vertices[i]
                    if CollisionPrimitives.circle_segment_collision(Circle(v, r), seg):
                        return True
                    segments.append(Segment(v, vertices[i+1]))
                segments.append(Segment(vertices[-1], vertices[0]))
                for s in segments:
                    if (CollisionPrimitives.circle_segment_collision(path_circles[0], s) or
                        CollisionPrimitives.circle_segment_collision(path_circles[1], s)):
                        return True
        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # Idea:
        #   1. for each segment: get distance between current point and nearest obstacle
        #   2. go up the path that far (until end of path)
        #   3. if at any point, distance < radius, add segment to list and go to next!

        # Note for TAs:
        # Not necessarily sampling based -> using principle of safety circle but since 
        # only the path + robot radius matters, the safety circle is reduced by radius r 
        # and the algorithm considers the path up to that point to be "safe" and samples 
        # at the boundary
        
        collided_segments = []

        path_segments = []
        for i in range(0, len(t.waypoints) - 1):
            path_segments.append(Segment(t.waypoints[i], t.waypoints[i+1]))

        polygons = []
        for obs in obstacles:
            if isinstance(obs, Triangle):
                vertices = [(obs.v1.x, obs.v1.y), (obs.v2.x, obs.v2.y), (obs.v3.x, obs.v3.y)]
            elif isinstance(obs, Polygon):
                vertices = []
                for v in obs.vertices:
                    vertices.append((v.x, v.y))
            elif isinstance(obs, Circle):
                vertices = [(obs.center.x, obs.center.y), (obs.center.x-r, obs.center.y), (obs.center.x, obs.center.y-r)]
                    
            polygons.append(shapely.geometry.Polygon(vertices))

        for i, seg in enumerate(path_segments):
            cur_point = shapely.geometry.Point(seg.p1.x, seg.p1.y)
            cont = True
            one_more = False
            theta = np.arctan2(seg.p2.y - seg.p1.y, seg.p2.x - seg.p1.x)

            seg_len = np.sqrt((seg.p2.y - seg.p1.y) ** 2 + (seg.p2.x - seg.p1.x) ** 2)
            cur_len = 0

            while cont:
                safety_dist = np.inf
                for j, obs in enumerate(obstacles):
                    if isinstance(obs, Circle):
                        cur_dist = cur_point.distance(shapely.geometry.Point(obs.center.x, obs.center.y))
                        cur_dist -= obs.radius
                    else:
                        cur_dist = cur_point.distance(polygons[j]) # distance w/ robot size
                    if cur_dist <= r:
                        collided_segments.append(i)
                        cont = False
                        break
                    elif cur_dist < safety_dist:
                        safety_dist = cur_dist
                
                if one_more:
                    break

                cur_len += safety_dist
                if cur_len >= seg_len:
                    cur_point = shapely.geometry.Point(seg.p2.x, seg.p2.y)
                    one_more = True
                else:
                    next_x = seg.p1.x + np.cos(theta) * safety_dist
                    next_y = seg.p1.y + np.sin(theta) * safety_dist
                    cur_point = shapely.geometry.Point(next_x, next_y)

        return collided_segments