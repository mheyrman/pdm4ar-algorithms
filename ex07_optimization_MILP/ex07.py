from typing import Tuple
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from pdm4ar.exercises_def.ex07.structures import (
    ProblemVoyage,
    OptimizationCost,
    Island,
    Constraints,
    Feasibility,
    SolutionVoyage,
)


def solve_optimization(problem: ProblemVoyage) -> SolutionVoyage:
    """
    Solve the optimization problem enforcing the requested constraints.

    Parameters
    ---
    problem : ProblemVoyage
        Contains the problem data: cost to optimize, starting crew, tuple of islands,
        and information about the requested constraint (the constraints not set to `None` +
        the `voyage_order` constraint)

    Returns
    ---
    out : SolutionVoyage
        Contains the feasibility status of the problem, and the optimal voyage plan
        as a list of ints if problem is feasible, else `None`.
    """

    # 5 Types of cost funtions (only 1 will be used):
    #     - Min nights to complete voyage
    #     - Max final crew size
    #     - Min total sailing time
    #     - Min L1 norm total travel distance
    #     - Minimize the maximum sailing time from 1 island to the next
    opt_cost = problem.optimization_cost
    # 6 Types of constraints (multiple can be applied):
    #     - Voyage order [omnipresent constraint] must start at 0, end at N-1, and visit N islands
    #     - Minimum number of nights that must be waited at an island before going to the next
    #     - Minimum crew size for the end of the journey
    #     - Maximum crew size for the end of the journey
    #     - Maximum sailing time from 1 island to the next
    #     - Maximum L1 norm sailing distance from 1 island to the next
    constraints = problem.constraints
    # An Island contains:
    #     - id: island id
    #     - arch: which archipelego the island belongs to
    #     - x, y: location of the island to calculate distance
    #     - departure: departure timetable
    #     - arrival: arrival timetable
    #     - nights: number of nights that must be spent to reset log poses
    #     - delta_crew: whether we gain Chopper (or maybe lost Usopp)
    isles = problem.islands
    # Start crew: integer representing number of crew members at start
    start_crew = problem.start_crew

    # get constraints for milp
    milp_constraints = check_constraints(constraints, start_crew, isles)

    # if opt_cost != OptimizationCost.min_total_travelled_L1_distance:
    #     raise(ValueError)

    # define cost function
    # [x_islands, distance_journeys, Z]
    solve_len = len(isles) + (isles[-1].arch) + 1
    c = np.zeros(solve_len)
    for i, isle in enumerate(isles):
        if opt_cost == OptimizationCost.min_total_nights:
            c[i] = isle.nights
        elif opt_cost == OptimizationCost.max_final_crew:
            c[i] = -isle.delta_crew
        elif opt_cost == OptimizationCost.min_total_sailing_time:
            # arrival - departure = time spent on an island more or less
            # arrival > departure
            # the longer spent, the smaller the value and the less time that can be spent travelling
            if i == 0:
                c[i] = -isle.departure
            elif isle == isles[-1]:
                c[i] = isle.arrival
            else:
                c[i] = isle.arrival - isle.departure

    # special convex cases:
    if opt_cost == OptimizationCost.min_total_travelled_L1_distance:
        # special constraint:
        milp_constraints = min_journey_L1_constraints(milp_constraints=milp_constraints, isles=isles)
        # need a way to represent the distance from start maybe?
        for i in range(len(isles), len(isles) + (isles[-1].arch)):
            c[i] = 1

    if opt_cost == OptimizationCost.min_max_sailing_time:
        milp_constraints = min_max_duration_constraints(milp_constraints=milp_constraints, isles=isles)
        c[-1] = 1

    integrality = np.ones_like(c)
    integrality[len(isles):] = 0

    bound_upper = np.ones(solve_len)
    bound_upper[len(isles):] = np.inf
    
    res = milp(c,
            integrality=integrality,
            bounds=Bounds(np.zeros(solve_len), bound_upper),
            constraints=milp_constraints)
    
    if res.success:
        # print("========================THE ONE PIECE IS REAL========================")
        feasibility = Feasibility.feasible
        voyage_plan = []
        for i, isle in enumerate(res.x):
            if np.allclose(1, isle):
                voyage_plan.append(i)
    else:
        feasibility = Feasibility.unfeasible
        voyage_plan = None

    return SolutionVoyage(feasibility, voyage_plan)

# additional constraints for min journey L1 cost
def min_journey_L1_constraints(milp_constraints: Constraints, isles: Tuple[Island]) -> Constraints:
    num_archs = isles[-1].arch + 1
    solve_len = len(isles) + (isles[-1].arch) + 1

    # 4 constraints to account for all linear combinations of ||d|| = |x2 - x1| + |y2 - y1|
    A_L1_0 = np.zeros((num_archs - 1, solve_len))
    A_L1_1 = np.zeros((num_archs - 1, solve_len))
    A_L1_2 = np.zeros((num_archs - 1, solve_len))
    A_L1_3 = np.zeros((num_archs - 1, solve_len))

    for i in range(0, np.shape(A_L1_0)[0]): # archipelago i, i+1
        for j in range(np.shape(A_L1_0)[1] - (num_archs)): # island 0 -> len(isles) - 1
            isle = isles[j]

            A_L1_0[i, j] = isle.x - isle.y if isle.arch == i else -isle.x + isle.y if isle.arch == i + 1 else 0
            A_L1_1[i, j] = isle.x + isle.y if isle.arch == i else -isle.x - isle.y if isle.arch == i + 1 else 0
            A_L1_2[i, j] = -isle.x - isle.y if isle.arch == i else isle.x + isle.y if isle.arch == i + 1 else 0
            A_L1_3[i, j] = -isle.x + isle.y if isle.arch == i else isle.x - isle.y if isle.arch == i + 1 else 0

        A_L1_0[i, len(isles) + i] = -1
        A_L1_1[i, len(isles) + i] = -1
        A_L1_2[i, len(isles) + i] = -1
        A_L1_3[i, len(isles) + i] = -1

    b_u_L1 = np.zeros(num_archs - 1)

    milp_constraints.extend([LinearConstraint(A=A_L1_0, ub=b_u_L1),
                            LinearConstraint(A=A_L1_1, ub=b_u_L1),
                            LinearConstraint(A=A_L1_2, ub=b_u_L1),
                            LinearConstraint(A=A_L1_3, ub=b_u_L1)])

    return milp_constraints

# additional constraint added for min_max_single_journey_duration cost
def min_max_duration_constraints(milp_constraints: Constraints, isles: Tuple[Island]) -> Constraints:
    solve_len = len(isles) + (isles[-1].arch) + 1

    num_archs = isles[-1].arch + 1 

    A_min_max = np.zeros((num_archs - 1, solve_len))

    for i in range(0, np.shape(A_min_max)[0]):
        for j in range(0, np.shape(A_min_max)[1] - (num_archs)):
            isle = isles[j]

            A_min_max[i, j] = -isle.departure if isle.arch == i else isle.arrival if isle.arch == i + 1 else 0

        A_min_max[i, -1] = -1
    
    b_u_min_max = np.zeros(num_archs - 1)

    milp_constraints.append(LinearConstraint(A=A_min_max, ub=b_u_min_max))

    return milp_constraints

# check constraints for going to the next island!
# contraints = matrix A in milp input
def check_constraints(input_constraints: Constraints, start_crew: int, isles: Tuple[Island]) -> Constraints:
    milp_constraints = []

    solve_len = len(isles) + (isles[-1].arch) + 1

    # # of archipelagos = last island.arch + 1 (since it starts at 0)
    num_archs = isles[-1].arch + 1

    A_order = np.zeros((num_archs, solve_len))
    A_min_nights = np.zeros((num_archs, solve_len))
    A_min_crew = np.zeros((num_archs, solve_len))
    A_max_crew = np.zeros((num_archs, solve_len))


    for i in range(np.shape(A_order)[0]): # num of archipelagos
        for j in range(np.shape(A_order)[1] - (num_archs)): # num of islands
            isle = isles[j]
            # voyage order -> always included!
            A_order[i, j] = 1 if isle.arch == i else 0

            # min nights
            if input_constraints.min_nights_individual_island:
                # let the lower bound do the work!
                A_min_nights[i, j] = isle.nights if isle.arch == i else 0
            # min crew size
            if input_constraints.min_total_crew:
                # if the current island archipelago is beyond the archipelago that we are checking, then set to 0
                A_min_crew[i, j] = isle.delta_crew if isle.arch <= i else 0
            # max crew size
            if input_constraints.max_total_crew:
                A_max_crew[i, j] = isle.delta_crew if isle.arch <= i else 0
    
    # defining upper and lower bounds
    b_l_order = np.ones(num_archs)
    b_u_order = np.ones(num_archs)
    milp_constraints.append(LinearConstraint(A=A_order, lb=b_l_order, ub=b_u_order))

    if input_constraints.min_nights_individual_island:
        b_l_min_nights = np.ones(num_archs) * input_constraints.min_nights_individual_island
        b_l_min_nights[0] = 0
        b_l_min_nights[-1] = 0
        milp_constraints.append(LinearConstraint(A=A_min_nights, lb=b_l_min_nights))
    if input_constraints.min_total_crew:
        b_l_min_crew = np.ones(num_archs) * (input_constraints.min_total_crew - start_crew)
        milp_constraints.append(LinearConstraint(A=A_min_crew, lb=b_l_min_crew))
    if input_constraints.max_total_crew:

        b_u_max_crew = np.ones(num_archs) * (input_constraints.max_total_crew - start_crew)
        milp_constraints.append(LinearConstraint(A=A_max_crew, ub=b_u_max_crew))

    # next two are special cases since they check the paths between islands:
    # max sailing time
    A_max_journey = np.zeros((num_archs - 1, solve_len))
    if input_constraints.max_duration_individual_journey:
        for i in range(np.shape(A_max_journey)[0]):
            for j in range(np.shape(A_max_journey)[1] - (num_archs)): # num of islands
                isle = isles[j]
                A_max_journey[i, j] = -isle.departure if isle.arch == i else isle.arrival if isle.arch == i + 1 else 0

        b_u_max_journey = np.ones(num_archs - 1) * input_constraints.max_duration_individual_journey
        milp_constraints.append(LinearConstraint(A=A_max_journey, ub=b_u_max_journey))

    # L1 requires 4 constraints: 1 for each possible combination of the value |x2-x1| + |y2-y1|
    A_max_L1_0 = np.zeros(((num_archs - 1), solve_len))
    A_max_L1_1 = np.zeros(((num_archs - 1), solve_len))
    A_max_L1_2 = np.zeros(((num_archs - 1), solve_len))
    A_max_L1_3 = np.zeros(((num_archs - 1), solve_len))

    if input_constraints.max_L1_distance_individual_journey:
        for i in range(0, np.shape(A_max_L1_0)[0]):
            for j in range(np.shape(A_max_L1_0)[1] - (num_archs)): # num of islands
                isle = isles[j]
                A_max_L1_0[i, j] = isle.x - isle.y if isle.arch == i else -isle.x + isle.y if isle.arch == i + 1 else 0
                A_max_L1_1[i, j] = isle.x + isle.y if isle.arch == i else -isle.x - isle.y if isle.arch == i + 1 else 0
                A_max_L1_2[i, j] = -isle.x - isle.y if isle.arch == i else isle.x + isle.y if isle.arch == i + 1 else 0
                A_max_L1_3[i, j] = -isle.x + isle.y if isle.arch == i else isle.x - isle.y if isle.arch == i + 1 else 0
                

        b_u_max_L1 = np.ones((num_archs - 1)) * input_constraints.max_L1_distance_individual_journey
        milp_constraints.extend([LinearConstraint(A=A_max_L1_0, ub=b_u_max_L1),
                                LinearConstraint(A=A_max_L1_1, ub=b_u_max_L1),
                                LinearConstraint(A=A_max_L1_2, ub=b_u_max_L1),
                                LinearConstraint(A=A_max_L1_3, ub=b_u_max_L1)])

    return milp_constraints # passes all constraint checks (except for constraints.min_total_crew)