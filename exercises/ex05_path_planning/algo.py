from typing import Sequence
import numpy as np

from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # given wheel_base, max_steering_angle, calculate the turning radius
    # TR = WB / tan(a)
    radius = DubinsParam(wheel_base / np.tan(max_steering_angle))

    return radius


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TurningCircle contains 2 Curve class objects (left and right)
    # Given SE2Transform, radius, using Curve.create_circle?

    cur_pos = current_config.p
    cur_theta = current_config.theta

    # center = pose + radius orthogonal to theta
    center_left = SE2Transform.identity()
    center_left.p[0] = cur_pos[0] + radius * np.cos(cur_theta + np.pi / 2.0)
    center_left.p[1] = cur_pos[1] + radius * np.sin(cur_theta + np.pi / 2.0)
    center_left.theta = 0.0

    left_circle = Curve.create_circle(center=center_left, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.LEFT, )

    center_right = SE2Transform.identity()
    center_right.p[0] = cur_pos[0] + radius * np.cos(cur_theta - np.pi / 2.0)
    center_right.p[1] = cur_pos[1] + radius * np.sin(cur_theta - np.pi / 2.0)
    center_right.theta = 0.0

    right_circle = Curve.create_circle(center=center_right, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.RIGHT)

    return TurningCircle(left=left_circle, right=right_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    lines = []

    # 2 EDGE CASES TO CHECK LATER: SMALL CIRCLE IS WITHIN LARGE CIRCLE COMPLETELY (NO 
    # TANGENT LINES), SMALL CIRCLE IS WITHIN LARGE CIRCLE AND THEY TOUCH (SAME IG)

    # Given 2 circles (!assume different radii for more challenge!), calculate a list of possible tangent lines
    # Must compute inner and outer tangents that connect 2 circles:

    # SHORTER VERSION
    # coords:
    radius = circle_start.radius

    start_x = circle_start.center.p[0]
    start_y = circle_start.center.p[1]
    end_x = circle_end.center.p[0]
    end_y = circle_end.center.p[1]

    cc_dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # OUTER TANGENTS
    if circle_start.type == circle_end.type:
        if circle_start.type == DubinsSegmentType.RIGHT and circle_end.type == DubinsSegmentType.RIGHT: # RR
            phi = np.arctan2(end_y - start_y, end_x - start_x) + (np.pi / 2.0)
        elif circle_start.type == DubinsSegmentType.LEFT and circle_end.type == DubinsSegmentType.LEFT: # LL
            phi = np.arctan2(end_y - start_y, end_x - start_x) - (np.pi / 2.0)

        tangent_start_x = start_x + radius * np.cos(phi)
        tangent_start_y = start_y + radius * np.sin(phi)

        tangent_end_x = end_x + radius * np.cos(phi)
        tangent_end_y = end_y + radius * np.sin(phi)

        theta = np.arctan2(tangent_end_y - tangent_start_y, tangent_end_x - tangent_start_x)
        
        start_config = SE2Transform.identity()
        start_config.p[0] = tangent_start_x
        start_config.p[1] = tangent_start_y
        start_config.theta = theta

        end_config = SE2Transform.identity()
        end_config.p[0] = tangent_end_x
        end_config.p[1] = tangent_end_y
        end_config.theta = theta

        line = Line(start_config=start_config, end_config=end_config)
        lines.append(line)
    else:
        out_rad = circle_start.radius + circle_end.radius
        if cc_dist == out_rad: # special case for kissing circles
            theta = np.arctan2(end_y - start_y, end_x - start_x)
            if circle_start.type == DubinsSegmentType.LEFT:
                theta += np.pi / 2.0
            else:
                theta -= np.pi / 2.0
            tangent_x = start_x + (end_x - start_x) / 2.0
            tangent_y = start_y + (end_y - start_y) / 2.0

            start_config = SE2Transform.identity()
            start_config.p[0] = tangent_x
            start_config.p[1] = tangent_y
            start_config.theta = theta

            end_config = SE2Transform.identity()
            end_config.p[0] = tangent_x
            end_config.p[1] = tangent_y
            end_config.theta = theta

            line = Line(start_config=start_config, end_config=end_config)
            lines.append(line)

        elif cc_dist > out_rad:
            if circle_start.type == DubinsSegmentType.LEFT:
                phi = np.arctan2(end_y - start_y, end_x -start_x) + np.arcsin(out_rad / cc_dist) - (np.pi / 2.0)
            else:
                phi = np.arctan2(end_y - start_y, end_x -start_x) - np.arcsin(out_rad / cc_dist) + (np.pi / 2.0)

            tangent_start_x = start_x + radius * np.cos(phi)
            tangent_start_y = start_y + radius * np.sin(phi)

            tangent_end_x = end_x + radius * np.cos(phi + np.pi)
            tangent_end_y = end_y + radius * np.sin(phi + np.pi)

            theta = np.arctan2(tangent_end_y - tangent_start_y, tangent_end_x - tangent_start_x)

            start_config = SE2Transform.identity()
            start_config.p[0] = tangent_start_x
            start_config.p[1] = tangent_start_y
            start_config.theta = theta

            end_config = SE2Transform.identity()
            end_config.p[0] = tangent_end_x
            end_config.p[1] = tangent_end_y
            end_config.theta = theta

            line = Line(start_config=start_config, end_config=end_config)
            lines.append(line)  

    return lines  # i.e., [Line(),...]

def calculate_arc_angle_length(start, end, circle_type, radius):
    arc_length = 0
    d_theta = 0
     # start circle length
    # counter-clockwise distance
    d_theta = ((end.theta % (2.0 * np.pi)) - (start.theta % (2.0 * np.pi))) % (np.pi * 2.0)
    if circle_type == 1: # clockwise distance
        d_theta = (2.0 * np.pi) - d_theta
    arc_length = np.pi * radius * (d_theta / np.pi)

    return d_theta, arc_length

def calculate_forward_path(start_config, end_config, radius, circle_combos):
    # Please keep segments with zero length in the return list & return a valid dubins path!
    # TEST CASES
    # start_config.p[0] = 0.0
    # start_config.p[1] = 0.0
    # start_config.theta = 0.0
 
    # end_config.p[0] = 0.0 * radius
    # end_config.p[1] = 0.0 * radius
    # end_config.theta = 0.0

    # possible combinations
    # let middle value 1/-1 indicate one side of the tangent and 2/-2 indicate the other
    start_circles = calculate_turning_circles(start_config, radius) # gives left and right circle 0,1
    end_circles = calculate_turning_circles(end_config, radius)
    mid_segment = []
    start_curves = []
    end_curves = []
    lengths = []
    for combo in circle_combos:
        if combo[0] == -1:
            start_circle = start_circles.left
        else:
            start_circle = start_circles.right

        if combo[2] == -1:
            end_circle = end_circles.left
        else:
            end_circle = end_circles.right

        cc_dist = np.sqrt((end_circle.center.p[1] - start_circle.center.p[1]) ** 2 + (end_circle.center.p[0] - start_circle.center.p[0]) ** 2)

        line_list = calculate_tangent_btw_circles(start_circle, end_circle)
        if line_list:
            cur_line = line_list[0]
            if combo[1] == 0: # straight line center segment
                start_curve = Curve(center=start_circle.center, start_config=start_config, end_config=cur_line.start_config, radius=start_circle.radius,
                            curve_type=start_circle.type)
                end_curve = Curve(center=end_circle.center, start_config=cur_line.end_config, end_config=end_config, radius=end_circle.radius,
                            curve_type=end_circle.type)

                # calculate length of path
                line_start = cur_line.start_config
                line_end = cur_line.end_config
                # line length
                line_length = np.sqrt((line_start.p[0] - line_end.p[0]) ** 2 + (line_start.p[1] - line_end.p[1]) ** 2)
                
                # start circle length
                # counter-clockwise distance
                d_theta_1, arc_length_1 = calculate_arc_angle_length(start_config, line_start, combo[0], radius)

                start_curve.length = arc_length_1
                start_curve.arc_angle = d_theta_1
                
                # end circle lengths
                # counter-clockwise distance
                d_theta_2, arc_length_2 = calculate_arc_angle_length(line_end, end_config, combo[2], radius)

                end_curve.length = arc_length_2
                end_curve.arc_angle = d_theta_2

                length = line_length + arc_length_1 + arc_length_2

                mid_segment.append(cur_line)
                start_curves.append(start_curve)
                end_curves.append(end_curve)
                lengths.append(length)

            elif cc_dist <= 4 * radius: # curve center segment is possible   
                centers = get_middle_circle_centers(start_circle=start_circle, end_circle=end_circle, cc_dist=cc_dist, radius=radius)
                if centers:
                    if combo[1] % 2 != 0: # intersection 0
                        pos = centers[0]
                    else: # intersection 1
                        pos = centers[1]

                    # get middle circle
                    mid_center = SE2Transform.identity()
                    mid_center.p[0] = pos[0]
                    mid_center.p[1] = pos[1]
                    mid_center.theta = 0.0
                    
                    # calculate points of contact
                    p_start = (start_circle.center.p[0] + (pos[0] - start_circle.center.p[0]) / 2.0,
                                start_circle.center.p[1] + (pos[1] - start_circle.center.p[1]) / 2.0)
                    p_end = (end_circle.center.p[0] + (pos[0] - end_circle.center.p[0]) / 2.0,
                                end_circle.center.p[1] + (pos[1] - end_circle.center.p[1]) / 2.0)

                    # calculate angles of contact (call it gamma)
                    gamma_start = np.arctan2(pos[1] - start_circle.center.p[1], pos[0] - start_circle.center.p[0])
                    gamma_end = np.arctan2(pos[1] - end_circle.center.p[1], pos[0] - end_circle.center.p[0])
                    if combo[1] > 0: # LRL
                        gamma_start = (gamma_start + np.pi / 2.0) % (2.0 * np.pi)
                        gamma_end = (gamma_end + np.pi / 2.0) % (2.0 * np.pi)
                    else:
                        gamma_start = (gamma_start - np.pi / 2.0) % (2.0 * np.pi)
                        gamma_end = (gamma_end - np.pi / 2.0)  % (2.0 * np.pi)

                    circle_start = SE2Transform.identity()
                    circle_start.p[0] = p_start[0]
                    circle_start.p[1] = p_start[1]
                    circle_start.theta = gamma_start

                    circle_end = SE2Transform.identity()
                    circle_end.p[0] = p_end[0]
                    circle_end.p[1] = p_end[1]
                    circle_end.theta = gamma_end

                    start_curve = Curve(center=start_circle.center, start_config=start_config, end_config=circle_start, radius=start_circle.radius,
                            curve_type=start_circle.type)
                    end_curve = Curve(center=end_circle.center, start_config=circle_end, end_config=end_config, radius=end_circle.radius,
                            curve_type=end_circle.type)

                    if combo[1] < 0:
                        mid_curve = Curve(center=mid_center, start_config=circle_start, end_config=circle_end, radius=radius,
                                curve_type=DubinsSegmentType.LEFT)
                    else:
                        mid_curve = Curve(center=mid_center, start_config=circle_start, end_config=circle_end, radius=radius,
                                curve_type=DubinsSegmentType.RIGHT)

                    # start circle length
                    # counter-clockwise distance
                    d_theta_1, arc_length_1 = calculate_arc_angle_length(start_config, circle_start, combo[0], radius)

                    start_curve.length = arc_length_1
                    start_curve.arc_angle = d_theta_1
                    
                    # end circle lengths
                    # counter-clockwise distance
                    d_theta_2, arc_length_2 = calculate_arc_angle_length(circle_end, end_config, combo[2], radius)

                    end_curve.length = arc_length_2
                    end_curve.arc_angle = d_theta_2

                    # mid circle lengths
                    # counter-clockwise distance
                    d_theta_3 = ((gamma_end % (2.0 * np.pi)) - (gamma_start % (2.0 * np.pi))) % (2.0 * np.pi)
                    if combo[1] > 0: # clockwise distance
                        d_theta_3 = (2.0 * np.pi) - d_theta_3
                    arc_length_3 = np.pi * radius * (d_theta_3 / np.pi)

                    mid_curve.length = arc_length_3
                    mid_curve.arc_angle = d_theta_3

                    length = arc_length_3 + arc_length_1 + arc_length_2

                    mid_segment.append(mid_curve)
                    start_curves.append(start_curve)
                    end_curves.append(end_curve)
                    lengths.append(length)

    if lengths:
        min_index = np.argmin(lengths)
        
        min_path = [start_curves[min_index], mid_segment[min_index], end_curves[min_index]]

        for segment in min_path:
            segment.gear = Gear.FORWARD
    else:
        min_path = []

    return min_path


def get_middle_circle_centers(start_circle: Curve, end_circle: Curve, cc_dist: float, radius: float) -> List[tuple]:
    centers = []

    # simplify circle center coords:
    start_x = start_circle.center.p[0]
    start_y = start_circle.center.p[1]
    start_rad = start_circle.radius + radius
    end_x = end_circle.center.p[0]
    end_y = end_circle.center.p[1]
    end_rad = end_circle.radius + radius

    if cc_dist >= np.abs(start_rad - end_rad) and cc_dist > 0:
        l = (start_rad ** 2 - end_rad ** 2 + cc_dist ** 2) / (2 * cc_dist)
        h = np.sqrt(start_rad ** 2 - l ** 2)

        x = start_x + l / cc_dist * (end_x - start_x) + h / cc_dist * (end_y - start_y)
        y = start_y + l / cc_dist * (end_y - start_y) - h / cc_dist * (end_x - start_x)
        centers.append((x, y))

        x = start_x + l / cc_dist * (end_x - start_x) - h / cc_dist * (end_y - start_y)
        y = start_y + l / cc_dist * (end_y - start_y) + h / cc_dist * (end_x - start_x)
        centers.append((x, y))

    return centers

def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    circle_combos = [(-1, 0, -1), (1, 0, -1), (-1, 0, 1), (1, 0, 1), (-1, 1, -1), (1, -1, 1), (-1, 2, -1), (1, -2, 1)]
        
    min_path = calculate_forward_path(start_config, end_config, radius, circle_combos)
    
    return min_path  # e.g., [Curve(), Line(),..]

def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # CALCULATE DUBINS
    dubins_path = calculate_dubins_path(start_config, end_config, radius)
    dubins_length = dubins_path[0].length + dubins_path[1].length + dubins_path[2].length

    shepp_combos = [(-1, 0, -1), (1, 0, -1), (-1, 0, 1), (1, 0, 1)]
    # CALCULATE REEDS_SHEPP
    shepp_path = calculate_forward_path(start_config=end_config, end_config=start_config, radius=radius, circle_combos=shepp_combos)
    shepp_length = shepp_path[0].length + shepp_path[1].length + shepp_path[2].length


    if shepp_length < dubins_length:
        shepp_path = shepp_path[::-1]
        for curve in shepp_path:
            curve_start = curve.start_config
            curve_end = curve.end_config

            curve.start_config = curve_end
            curve.end_config = curve_start

            curve.gear = Gear.REVERSE

        return shepp_path
    return dubins_path