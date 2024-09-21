import numpy as np
import triangle as tr

from pdm4ar.exercises_def.ex06.structures import *


def triangle_area(v1: Point, v2: Point, v3: Point):
    return np.abs(v1.x * (v2.y - v3.y) + v2.x * (v3.y- v1.y) + v3.x * (v1.y - v2.y))

def get_triangle_orientation(v1: Point, v2: Point, v3: Point): # p q r
    orientation = (v2.y - v1.y) * (v3.x - v2.x) - (v3.y - v2.y) * (v2.x - v1.x)

    if orientation > 0: # clockwise
        return 1
    elif orientation < 0: # counter-clockwise
        return -1
    return 0 # straight line

class CollisionPrimitives:
    """
    Class of collusion primitives
    """
    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        # find distance between point and circle center
        # if distance > radius: no collision!

        center_x = c.center.x
        center_y = c.center.y
        radius = c.radius

        dist = np.sqrt((p.y - center_y) ** 2 + (p.x - center_x) ** 2)
        if radius < dist:
            return False
        return True

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        # find distance between point and all 3 triangle vertices
        v = [t.v1, t.v2, t.v3]

        A = round(triangle_area(v[0], v[1], v[2]), 12)
        A_2 = round(triangle_area(p, v[0], v[1]) + triangle_area(p, v[1], v[2]) + triangle_area(p, v[0], v[2]), 12)

        if A == A_2:
            return True
        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        poly_vertices = []
        for b in poly.vertices:
            poly_vertices.append((b.x, b.y))
        
        A = dict(vertices=np.array(poly_vertices))
        
        triangles = tr.triangulate(A)

        verts = triangles['vertices']
        vert_combos = triangles['triangles']

        is_in = []

        # for every possible combination of vertices
        for i in range(vert_combos.shape[0]):
            tv = vert_combos[i] # triangle vertices
            v1 = Point(verts[tv[0]][0], verts[tv[0]][1])
            v2 = Point(verts[tv[1]][0], verts[tv[1]][1])
            v3 = Point(verts[tv[2]][0], verts[tv[2]][1])

            # create a triangle
            t = Triangle(v1, v2, v3)

            # check if the point is within the triangle
            is_in.append(CollisionPrimitives.triangle_point_collision(t, p))

        # if the point is within any formed triangles, then it is within the polygon
        if np.any(is_in):
            return True
        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        # find perpendicular distance between center of circle and line
        # if <= radius : collision
        center = c.center

        p1 = segment.p1
        p2 = segment.p2

        # use alpha, beta, gamma instead of abc to protect c: Circle
        alpha = p1.y - p2.y
        beta = p2.x - p1.x
        gamma = (p1.x - p2.x) * p1.y + (p2.y - p1.y) * p1.x

        dist = np.abs(center.x * alpha + center.y * beta + gamma) / np.sqrt(alpha ** 2 + beta ** 2)

        if dist <= c.radius:
            # check if the closest perpendicular point is actually between the segment
            det = alpha ** 2 + beta ** 2
            m = (-alpha * (center.y - p1.y) + beta * (center.x - p1.x)) / det

            intersection_x = p1.x + m * beta
            intersection_y = p1.y - m * alpha

            if ((CollisionPrimitives.circle_point_collision(c, p1) or CollisionPrimitives.circle_point_collision(c, p2)) or
                ((p1.x <= intersection_x <= p2.x or p2.x <= intersection_x <= p1.x) and
                (p1.y <= intersection_y <= p2.y or p2.y <= intersection_y <= p1.y))):
                return True
        return False

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        # check if any segment of the triangle intersects with the segment
        # this can be done with some fancy maths using orientations

        vertices = [t.v1, t.v2, t.v3]

        combos = [(0, 1), (1, 2), (0, 2)]

        for combo in combos:
            v1 = vertices[combo[0]]
            v2 = vertices[combo[1]]

            if ((CollisionPrimitives.triangle_point_collision(t, segment.p1) or CollisionPrimitives.triangle_point_collision(t, segment.p2)) or
                ((get_triangle_orientation(v1, v2, segment.p1) != get_triangle_orientation(v1, v2, segment.p2)) and
                (get_triangle_orientation(segment.p1, segment.p2, v1) != get_triangle_orientation(segment.p1, segment.p2, v2)))): 
                return True

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        poly_vertices = []
        for b in p.vertices:
            poly_vertices.append((b.x, b.y))
        
        A = dict(vertices=np.array(poly_vertices))
        
        triangles = tr.triangulate(A)

        verts = triangles['vertices']
        vert_combos = triangles['triangles']

        is_in = []

        # for every possible combination of vertices
        for i in range(vert_combos.shape[0]):
            tv = vert_combos[i] # triangle vertices
            v1 = Point(verts[tv[0]][0], verts[tv[0]][1])
            v2 = Point(verts[tv[1]][0], verts[tv[1]][1])
            v3 = Point(verts[tv[2]][0], verts[tv[2]][1])

            # create a triangle
            t = Triangle(v1, v2, v3)

            if ((CollisionPrimitives.polygon_point_collision(p, segment.p1) or CollisionPrimitives.polygon_point_collision(p, segment.p2)) or
                CollisionPrimitives.triangle_segment_collision(t, segment)):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        # create AABB or bounding sphere

        p_min, p_max = p.get_boundaries()
        
        v = [Point(p_min.x, p_max.y), Point(p_max.x, p_max.y), Point(p_max.x, p_min.y), Point(p_min.x, p_min.y)]

        t1 = Triangle(v[0], v[1], v[2])
        t2 = Triangle(v[1], v[2], v[3])

        if (CollisionPrimitives.triangle_segment_collision(t1, segment) or 
            CollisionPrimitives.triangle_segment_collision(t2, segment)):
            if (CollisionPrimitives.polygon_segment_collision(p, segment)):
                return True

        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks        

        return AABB(p_min=Point(0, 0), p_max=Point(0, 0))

    @staticmethod
    def circle_triangle_collision(c: Circle, t: Triangle) -> bool:
        if CollisionPrimitives.triangle_point_collision(t, c.center):
            return True
        
        segs = [Segment(t.v1, t.v2), Segment(t.v2, t.v3), Segment(t.v3, t.v1)]
        for seg in segs:
            if CollisionPrimitives.circle_segment_collision(c, seg):
                return True

        p = Polygon([t.v1, t.v2, t.v3])

        return CollisionPrimitives.circle_polygon_collision(c, p)
        

    @staticmethod
    def circle_polygon_collision(c: Circle, p: Polygon) -> bool:
        if CollisionPrimitives.polygon_point_collision(p, c.center):
            return True
        
        vertices = p.vertices
        for i in range(0, len(vertices) - 1):
            seg = Segment(vertices[i], vertices[i+1])
            if CollisionPrimitives.circle_segment_collision(c, seg):
                return True
        seg = Segment(vertices[-1], vertices[0])
        if CollisionPrimitives.circle_segment_collision(c, seg):
            return True

        return False

    @staticmethod
    def circle_circle_collision(c1: Circle, c2: Circle) -> bool:
        dist_cc = np.sqrt((c1.center.y - c2.center.y) ** 2 + (c1.center.x - c2.center.x) ** 2)
        if (c1.radius + c2.radius) > dist_cc:
            return True
        return False
