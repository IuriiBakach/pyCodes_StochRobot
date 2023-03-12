from scipy.stats import gamma
from classes_needed import *
import copy
import random
from random import randint
import numpy as np


def all_distances(depot_loc, list_of_cust, inner_zone_coords):
    """
        Compute the matrix of distances with corresponding number of blocks for each of the 4 possible routes for each
    customer involved

    :param depot_loc:  a [x,y] set of depot coords [1.2,1.7]
    :param list_of_cust: a list of customers considered
    :param inner_zone_coords: array of vertices of the inner zone of this form [xLow, yLow, xUp, yUp] [.3, .6, 1.7, 1.3]
    :return: the matrix of distances
    """

    distance_matrix = np.zeros((len(list_of_cust), 4), dtype='f,f').tolist()
    tmp = []

    # for every customer compute distances
    for index, cust in enumerate(list_of_cust):
        # if customer falls into case 1, i.e. same upper zone with the depot, need to compute Manh distances and
        # fill all 4 cells with the same values. check this
        if cust.yCoord >= inner_zone_coords[3] or cust.xCoord <= inner_zone_coords[0] or cust.xCoord >= \
                inner_zone_coords[2]:
            # compute manhattan distance
            case_distance = (abs(cust.xCoord - depot_loc[0]) + abs(cust.yCoord - depot_loc[1]), 0)
            # modify the whole row so that all the distances are the same for 4 possible routes

            tmp = [case_distance, (1000, 1000), (1000, 1000), (1000, 1000)]
            # check distance matrix indices
            distance_matrix[index] = tmp
            tmp = []

        # if customer falls into case 3, i.e. is in the lower part of the graph directly below the zone.
        # in this case we have 2 possible routes:
        # 1) shortest through the zone
        # 2) shortest avoiding the zone
        # other 2 routes should be filled with something (perhaps inf?)
        if inner_zone_coords[0] < cust.xCoord < inner_zone_coords[2] and \
                cust.yCoord <= inner_zone_coords[1]:
            # 1) shortest through the zone, (blocks outside inner zone, blocks inside inner zone)
            case1_distance = (abs(cust.xCoord - depot_loc[0]) + abs(cust.yCoord - depot_loc[1]) -
                              (inner_zone_coords[3] - inner_zone_coords[1]),
                              inner_zone_coords[3] - inner_zone_coords[1])
            # 2) shortest avoiding the zone
            case2_distance_a = abs(inner_zone_coords[0] - depot_loc[0]) + abs(inner_zone_coords[1] - depot_loc[1]) + \
                               abs(inner_zone_coords[0] - cust.xCoord) + abs(
                inner_zone_coords[1] - cust.yCoord)

            case2_distance_b = abs(inner_zone_coords[2] - depot_loc[0]) + abs(
                inner_zone_coords[1] - depot_loc[1]) + abs(inner_zone_coords[2] - cust.xCoord) + abs(
                inner_zone_coords[1] - cust.yCoord)

            # fill the proper row that corresponds to a customer
            case2_distance = (min(case2_distance_a, case2_distance_b), 0)
            tmp = [case1_distance, case2_distance, (1000, 1000), (1000, 1000)]

            # check distance matrix indices
            distance_matrix[index] = tmp
            tmp = []

        # if customer falls into case 2, i.e. has coord in the zone, then 4 different paths are possible.
        # this else should be fixed
        elif inner_zone_coords[0] < cust.xCoord < inner_zone_coords[2] and \
                inner_zone_coords[1] < cust.yCoord < inner_zone_coords[3]:
            # step 1: find 4 entry points: direct distances from a customer to a edges and these are inner distances
            # points: 1-left, 2-down, 3-up, 4-right
            # a point contain coords and number of blocks inside a zone

            entry_pt_1 = [(inner_zone_coords[0], cust.yCoord), abs(inner_zone_coords[0] - cust.xCoord)]
            entry_pt_2 = [(cust.xCoord, inner_zone_coords[1]), abs(inner_zone_coords[1] - cust.yCoord)]
            entry_pt_3 = [(cust.xCoord, inner_zone_coords[3]), abs(inner_zone_coords[3] - cust.yCoord)]
            entry_pt_4 = [(inner_zone_coords[2], cust.yCoord), abs(inner_zone_coords[2] - cust.xCoord)]

            # step 2: find the distance to all entry points avoiding inner zone
            # to points 1, 3, 4 it's easy; to point 2 need to use same approach as for the case 3 -> min(A, B)

            case2_distance_1 = (abs(depot_loc[0] - entry_pt_1[0][0]) +
                                abs(depot_loc[1] - entry_pt_1[0][1]), entry_pt_1[1])
            case2_distance_3 = (abs(depot_loc[0] - entry_pt_3[0][0]) +
                                abs(depot_loc[1] - entry_pt_3[0][1]), entry_pt_3[1])
            case2_distance_4 = (abs(depot_loc[0] - entry_pt_4[0][0]) +
                                abs(depot_loc[1] - entry_pt_4[0][1]), entry_pt_4[1])

            # left bottom corner and right bottom corner distance
            case2_distance_2a = abs(depot_loc[0] - inner_zone_coords[0]) + abs(depot_loc[1] - inner_zone_coords[1]) + \
                                abs(entry_pt_2[0][0] - inner_zone_coords[0])
            case2_distance_2b = abs(depot_loc[0] - inner_zone_coords[2]) + abs(depot_loc[1] - inner_zone_coords[1]) + \
                                abs(entry_pt_2[0][0] - inner_zone_coords[2])
            case2_distance_2 = (min(case2_distance_2a, case2_distance_2b), entry_pt_2[1])

            tmp = [case2_distance_1, case2_distance_2, case2_distance_3, case2_distance_4]
            distance_matrix[index] = tmp
            tmp = []

    return distance_matrix


def dist_matr_trim(distance_matrix_raw, los_matrix, cust_list):
    """
    This function has to take in a distance matrix, then for every customer find the best path in terms of expected
    travel time. The output is an array of tuples of the form (distance, res_shape) along with path indices

    no need to keep scale par since it's always the same and initially given

    :param distance_matrix_raw: matrix of all distances
    :param los_matrix: pedestrian intensity los for two zones
    :param cust_list: list of all customers
    :return: a tuple of travel distance and shape coeff's with path id's
    """

    # how do I compute the resulting shape? dist_outer*shape_outer+dist_inner*shape_inner/dist_outer+dist_inner

    # service time is 2 minutes or 1/30 hour
    service_time = 2

    # setup a set of needed shape and scale parameters
    shape_zone_out = los_matrix[0][0][0]
    shape_zone_in = los_matrix[0][1][0]
    scale_par = los_matrix[0][0][1]

    # I think I need to change the robot speed too
    robot_speed = 3

    # create an empty distance matrix and index matrix
    trimmed_matr_dist = np.zeros((1, len(cust_list) + 1))
    trimmed_matr_shape = np.zeros((1, len(cust_list) + 1))

    best_path_indices = [0]

    best_paths = [0]

    best_path_time = 999999
    curr_path_distance = 999999
    # go through all paths for customers
    for index_upp, elem in enumerate(distance_matrix_raw):
        # go through every path
        for index, path in enumerate(elem):
            # compute an expected travel time of a path including service time
            '''I need to multiply by 60 in order to get to minutes conversion'''
            curr_path_time = ((path[0] * shape_zone_out + path[1] * shape_zone_in) / robot_speed) * 60 + service_time
            # compute distance
            curr_path_distance = path[0] + path[1]
            # compute resulting shape coeff in the way it takes into account scale parameter too
            curr_res_shape = curr_path_time / (curr_path_distance * scale_par)

            if curr_path_time < best_path_time:
                # keep track of best results for the customer
                best_path_time = curr_path_time
                best_path_distance = curr_path_distance
                best_res_shape = curr_res_shape

                best_path_time_index = index

                # need to check during the debugging phase if I need this best path part
                best_path = path

        # add best distance and best shape coeff to the matrix
        trimmed_matr_dist[0][index_upp + 1] = best_path_distance
        trimmed_matr_shape[0][index_upp + 1] = best_res_shape

        # update the outputs
        best_path_indices.append(best_path_time_index)
        best_paths.append(best_path)

        best_path_time = 999999
        curr_path_distance = 999999

    # multiply by 60 in order to get in minutes
    # trimmed_matr[0][1] = trimmed_matr[0][1]

    trimmed_matr_dist[0][0] = 0
    trimmed_matr_shape[0][0] = 0

    # trimmed_matr returns all the corresponding distances and shape values for the best paths for all custs.
    # 0-position is depot

    return trimmed_matr_dist, trimmed_matr_shape, best_path_indices, best_paths


def expected_delay(shape, scale_par, uppertw):
    """
        This function computes expected delay of the arrival of robot r to the customer c taking into account
    all previously visited customers

    input parameters:
    shape- shape parameter obtained based on distance traveled till this customer,
    corresponds to _jv in the paper
    scale- scale parameter (fixed for all customers in the specific run)
    upperTW- late time window for a customer
    :return: expected delay
    """

    # ______ create required gamma distributions 3 modify initial creation of the distributions
    gd1 = gamma(shape, scale=scale_par)
    gd2 = gamma(shape + 1, scale=scale_par)
    ans = shape * scale_par * (1 - gd2.cdf(uppertw)) - uppertw * (1 - gd1.cdf(uppertw))

    return ans


def expected_earliness(shape, scale_par, lowertw):
    """
        This function computes expected earliness of the arrival of robot r to the customer c taking into account
    all previously visited customers

    input parameters:
        shape- shape parameter obtained as distance (travel time) traveled till this customer
        scale- scale parameter (fixed for all customers in the specific run)
        upperTW- late time window for a customer

    :return: expected earliness
    """

    gd1 = gamma(shape, scale=scale_par)
    gd2 = gamma(shape + 1, scale=scale_par)

    ans = lowertw * gd1.cdf(lowertw) - shape * scale_par * gd2.cdf(lowertw)

    return ans


def earliness_array(precomputed_distances, customers_in_route, shape, scale):
    """
    This function recomputes earliness for all customers based on the current route

    :param precomputed_distances: distances (time) to all current customers given all previous are visited
    :param customers_in_route: list of customer in the route
    :param shape: shape parameter for gamma distribution
    :param scale: scale parameter for gamma distribution
    :return: a list of computed earliness'
    """
    # if I am to consider different alphas for different customers this and next function should be rewritten in
    # some way

    earliness = [0] * len(precomputed_distances)
    for i, elem in enumerate(precomputed_distances[1:], 1):
        earliness[i] = expected_earliness(shape * elem, scale, customers_in_route[i].getEarlyTW())
    return earliness


def lateness_array(precomputed_distances, customers_in_route, shape, scale):
    """
    This function recomputes lateness for all customers based on the current route

    :param precomputed_distances: distances (time) to all current customers given all previous are visited
    :param customers_in_route: list of customer in the route
    :param shape: shape parameter for gamma distribution
    :param scale: scale parameter for gamma distribution
    :return: a list of computed lateness'
    """
    lateness = [0] * len(precomputed_distances)
    for i, elem in enumerate(precomputed_distances[1:], 1):
        lateness[i] = expected_delay(shape * elem, scale, customers_in_route[i].getLateTW())
    return lateness


def earliness_array_v_2(cust_ordering, to_cust_distances, to_cust_shape, scale):
    """
    This is an updated function of an earliness array. The idea is to take inputs in the form of the cust_distance,
    shape parameters and return earliness for all customers given the accumulated manner of the expected arrival time.

    This computation is performed for 1 current route only

    :param cust_ordering: order of customers in the route. TYPE = list with cust's based on positions
    :param to_cust_distances: distances to customers from a hub. TYPE = list with corresponding distances
    :param to_cust_shape: shape parameter to be used together with distances. Essentially, the product of distance and
            shape is the shape to use in the expected_earliness function
    :param scale: scale parameter for gamma distribution
    :return: a list of computed earliness'
    """

    # step 1: I need to get \alpha_jv, a shape of arrival to custs.

    # check if cust ordering has a depot as 0-cust

    # this array contains arrival times to customers in the form of \alpha_jv
    exp_arrival_shape = [0] * len(cust_ordering)

    # compute all the expected shape of arrivals
    for index, elem in enumerate(exp_arrival_shape):
        if index == 0:
            exp_arrival_shape[index] = to_cust_distances[index] * to_cust_shape[index]
        else:
            # I need to not just add a previous exp arrival time, but also the time to return
            exp_arrival_shape[index] = exp_arrival_shape[index - 1] + to_cust_distances[index - 1] * to_cust_shape[
                index - 1] + to_cust_distances[index] * to_cust_shape[index]

    # step 2: once I have them, I can send each of them into the expected_earliness function
    earliness = [0] * len(cust_ordering)

    # start with the first element and then afterwards add 0 in the beginning

    exp_arrival_shape = exp_arrival_shape[1:]

    for index, elem in enumerate(exp_arrival_shape, 1):
        earliness_elem = expected_earliness(elem, scale, cust_ordering[index].getEarlyTW())
        earliness[index] = earliness_elem

    # insert 0's because of the depot
    exp_arrival_shape.insert(0, 0)

    return earliness


def lateness_array_v_2(cust_ordering, to_cust_distances, to_cust_shape, scale):
    """
    This is an updated function of a lateness array. The idea is to take inputs in the form of the cust_distance,
    shape parameters and return lateness for all customers given the accumulated manner of the expected arrival time.

    This computation is performed for 1 current route only

    :param cust_ordering: order of customers in the route. TYPE = list with cust's based on positions
    :param to_cust_distances: distances to customers from a hub. TYPE = list with corresponding distances
    :param to_cust_shape: shape parameter to be used together with distances. Essentially, the product of distance and
            shape is the shape to use in the expected_earliness function
    :param scale: scale parameter for gamma distribution
    :return: a list of computed earliness'
    """

    # step 1: I need to get \alpha_jv, a shape of arrival to custs.

    # check if cust ordering has a depot as 0-cust

    # this array contains arrival times to customers in the form of \alpha_jv
    exp_arrival_shape = [0] * len(cust_ordering)

    # compute all the expected shape of arrivals
    for index, elem in enumerate(exp_arrival_shape):
        if index == 0:
            exp_arrival_shape[index] = to_cust_distances[index] * to_cust_shape[index]
        else:
            # I need to not just add a previous exp arrival time, but also the time to return
            exp_arrival_shape[index] = exp_arrival_shape[index - 1] + to_cust_distances[index - 1] * to_cust_shape[
                index - 1] + to_cust_distances[index] * to_cust_shape[index]

    # step 2: once I have them, I can send each of them into the expected_earliness function

    lateness = [0] * len(cust_ordering)

    # start with the first element and then afterwards add 0 in the beginning

    exp_arrival_shape = exp_arrival_shape[1:]

    for index, elem in enumerate(exp_arrival_shape, 1):
        lateness_elem = expected_delay(elem, scale, cust_ordering[index].getLateTW())
        lateness[index] = lateness_elem
    # insert 0's because of the depot

    exp_arrival_shape.insert(0, 0)

    return lateness


def one_shift(cust1, cust2, route_plan, distMatr, shape, scale):
    """
    Execute a 1-shift (vertex reassignment) operator. That is remove first customer form a current route
    and insert into the route after the second customer
    :param cust1: customer to relocate
    :param cust2: customer to insert cust1 after
    :param route_plan: an initial set of routes
    :param distMatr: matrix of distances form depot to customers
    :param shape: shape parameter for gamma distribution
    :param scale: scale parameter for gamma distribution

    :return: change in objective function after this operation

    """

    # the procedure is the following: find customer 1 in the route_plan and remove this customer from it.
    # then find the second customer and insert the first one after it. Report the change in the objective function

    total_earl = 0
    total_late = 0

    route_plan_copy = copy.deepcopy(route_plan)

    # find the location of the first customer in the route and remove it from the route if the customer is present in
    # in that route
    for route in route_plan_copy:
        position_cust_one = route.find_customer_pos(cust1)
        if type(position_cust_one) is int:
            route.remove_customer(position_cust_one, distMatr, shape, scale)
            break

    # find the location of the second customer and insert customer 1 after it
    for route in route_plan_copy:
        position_cust_two = route.find_customer_pos(cust2)
        if type(position_cust_two) is int:
            route.insert_customer(position_cust_two + 1, cust1, distMatr, shape, scale)
            break

    # after the insertion has been completed, report the modified combined objective function

    for route in route_plan_copy:
        total_earl += route.total_earliness()
        total_late += route.total_lateness()

    # return modified route_plan and corresponding sum of the objective function

    return [(cust1, cust2), total_earl + total_late, route_plan_copy]  # [route_plan, total_earl + total_late]


def one_shift_v2(cust1, cust2, route_plan, distMatr, shapes, scale):
    """
    Execute a 1-shift (vertex reassignment) operator. That is remove first customer form a current route
    and insert into the route after the second customer
    :param cust1: customer to relocate
    :param cust2: customer to insert cust1 after
    :param route_plan: an initial set of routes
    :param distMatr: matrix of distances form depot to customers
    :param shapes: shape parameters for best paths
    :param scale: scale parameter for gamma distribution

    :return: change in objective function after this operation

    """

    # the procedure is the following: find customer 1 in the route_plan and remove this customer from it.
    # then find the second customer and insert the first one after it. Report the change in the objective function

    total_earl = 0
    total_late = 0

    route_plan_copy = copy.deepcopy(route_plan)

    # find the location of the first customer in the route and remove it from the route if the customer is present in
    # in that route
    for route in route_plan_copy:
        position_cust_one = route.find_customer_pos(cust1)
        if type(position_cust_one) is int:
            # route.remove_customer(position_cust_one, distMatr, shape, scale)
            route.remove_customer_v_2(position_cust_one, scale)
            break

    # find the location of the second customer and insert customer 1 after it
    for route in route_plan_copy:
        position_cust_two = route.find_customer_pos(cust2)
        if type(position_cust_two) is int:
            # route.insert_customer(position_cust_two + 1, cust1, distMatr, shape, scale)
            route.insert_customer_v_2(position_cust_two + 1, cust1, distMatr, shapes, scale)
            break

    # after the insertion has been completed, report the modified combined objective function

    for route in route_plan_copy:
        total_earl += route.total_earliness()
        total_late += route.total_lateness()

    # return modified route_plan and corresponding sum of the objective function

    return [(cust1, cust2), total_earl + total_late, route_plan_copy]  # [route_plan, total_earl + total_late]


def exchange(cust1, cust2, route_plan, distMatr, shape, scale):
    """
    This operation switches customer 1 and customer 2 in routes

    :param cust1: customer to relocate 1
    :param cust2: customer to relocate 2
    :param route_plan: an initial set of routes
    :param distMatr: matrix of distances form depot to customers
    :param shape: shape parameter for gamma distribution
    :param scale: scale parameter for gamma distribution
    :return: change in objective function after this operation
    """

    route_plan_copy = copy.deepcopy(route_plan)

    total_earl = 0
    total_late = 0

    # find the locations of these 2 customers
    for route in route_plan_copy:
        position_cust_one = route.find_customer_pos(cust1)
        if type(position_cust_one) is int:
            loc_cust_one = [route.id, position_cust_one]
            break

    for route in route_plan_copy:
        position_cust_two = route.find_customer_pos(cust2)
        if type(position_cust_two) is int:
            loc_cust_two = [route.id, position_cust_two]
            break

    # the next step is to remove those customers from routes, insert into new positions and return resulting routes
    # and obj fcn

    # the problem arises when I remove customers from the same route. Need to keep track of the positions of removed
    # customers and insert into the proper positions afterwards. I.e. modify the next 4 lines accordingly

    '''
    insert customer 2 after customer 1, remove customer 1; insert customer 1 after customer 2, remove customer 2
    '''

    # insert customer has an error
    route_plan_copy[loc_cust_one[0]].insert_customer(position_cust_one, cust2, distMatr, shape, scale)
    route_plan_copy[loc_cust_one[0]].remove_customer(position_cust_one + 1, distMatr, shape, scale)

    route_plan_copy[loc_cust_two[0]].insert_customer(position_cust_two, cust1, distMatr, shape, scale)
    route_plan_copy[loc_cust_two[0]].remove_customer(position_cust_two + 1, distMatr, shape, scale)

    for route in route_plan_copy:
        total_earl += route.total_earliness()
        total_late += route.total_lateness()

    return [(cust1, cust2), total_earl + total_late, route_plan_copy]


def exchange_v2(cust1, cust2, route_plan, distMatr, shapes, scale):
    """
    This operation switches customer 1 and customer 2 in routes

    :param cust1: customer to relocate 1
    :param cust2: customer to relocate 2
    :param route_plan: an initial set of routes
    :param distMatr: matrix of distances form depot to customers
    :param shapes: shape parameter for gamma distribution
    :param scale: scale parameter for gamma distribution
    :return: change in objective function after this operation
    """

    route_plan_copy = copy.deepcopy(route_plan)

    total_earl = 0
    total_late = 0

    # find the locations of these 2 customers
    for route in route_plan_copy:
        position_cust_one = route.find_customer_pos(cust1)
        if type(position_cust_one) is int:
            loc_cust_one = [route.id, position_cust_one]
            break

    for route in route_plan_copy:
        position_cust_two = route.find_customer_pos(cust2)
        if type(position_cust_two) is int:
            loc_cust_two = [route.id, position_cust_two]
            break

    # the next step is to remove those customers from routes, insert into new positions and return resulting routes
    # and obj fcn

    # the problem arises when I remove customers from the same route. Need to keep track of the positions of removed
    # customers and insert into the proper positions afterwards. I.e. modify the next 4 lines accordingly

    '''
    insert customer 2 after customer 1, remove customer 1; insert customer 1 after customer 2, remove customer 2
    '''

    route_plan_copy[loc_cust_one[0]].insert_customer_v_2(position_cust_one, cust2, distMatr, shapes, scale)
    route_plan_copy[loc_cust_one[0]].remove_customer_v_2(position_cust_one + 1, scale)

    route_plan_copy[loc_cust_two[0]].insert_customer_v_2(position_cust_two, cust1, distMatr, shapes, scale)
    route_plan_copy[loc_cust_two[0]].remove_customer_v_2(position_cust_two + 1, scale)

    for route in route_plan_copy:
        total_earl += route.total_earliness()
        total_late += route.total_lateness()

    return [(cust1, cust2), total_earl + total_late, route_plan_copy]


def total_obj_fcn_route(routePlan):
    """
    This function computes the objective function for the list of routes

    :param routePlan: list of routes
    :return: a sum of total earliness and total lateness
    """
    total_earl = 0
    total_late = 0
    for route in routePlan:
        total_earl += route.total_earliness()
        total_late += route.total_lateness()
    return total_earl + total_late


def create_cand_list(customerList, currentSolutionRoutes, max_cand_list_len, distances, shapes, scale):
    """
    Create a candidate list of the len = max_cand_list_len for the tabu search

    :param customerList: initial customer list
    :param currentSolutionRoutes: current solution
    :param max_cand_list_len: how many candidates to create
    :param distances: matrix of distance between customers
    :param shapes: list of shapes for best paths
    :param scale: scale
    :return: a sorted list of possible moves
    """
    candidate_list = []

    while len(candidate_list) < max_cand_list_len:  # this works nicely
        # select vertices randomly
        # here I get the INDEX of the customers I'd like to use in the swap
        random_sample = random.sample(range(0, len(customerList)), 2)
        r_cust_one = random_sample[0]
        r_cust_two = random_sample[1]

        # once I selected these indices, I need to randomly chose an operation
        r_operation = randint(1, 2)
        if r_operation == 1:
            # I need to perform a one_shift operation
            # operation has [(cust1, cust2), obj_fcn, modifiedRoute]
            # listOfRoutes is the current solution

            # also need to check if this works after deepcopy!!!

            operation = one_shift_v2(customerList[r_cust_one - 1], customerList[r_cust_two - 1],
                                     currentSolutionRoutes[0],
                                     distances, shapes, scale)
            # operation is [(cust1, cust2), total_earl + total_late, route_plan_copy]

            # apparently I need to check the result of operation, run shifting here and record that result here
            # create tmp var to get shiftings updates and rewrite operation afterwards

            #by_cust_shift_per_route, tmp_obj_value_after_fwd_shift, percent_early, percent_late = forward_shifting(
            #    operation[2], scale, 5)
            #operation[1] = tmp_obj_value_after_fwd_shift

        else:
            # perform exchange operation
            operation = exchange_v2(customerList[r_cust_one - 1], customerList[r_cust_two - 1],
                                    currentSolutionRoutes[0],
                                    distances, shapes, scale)

            #by_cust_shift_per_route, tmp_obj_value_after_fwd_shift, percent_early, percent_late = forward_shifting(
            #    operation[2], scale, 5)
            #operation[1] = tmp_obj_value_after_fwd_shift

            # operation is [(cust1, cust2), total_earl + total_late, route_plan_copy]

        # I need to add to a candidate list neighborhood and operation id (either 1 or 2)
        candidate_list.append([operation, r_operation])
    # Sorted candidate list based on the changed objective function
    # now I can take first element of whatever I need
    candidate_list.sort(key=lambda x: x[0][1])

    return candidate_list


def tabu_search(custList, matrOfDistances, listOfRoutes, shapes, scale):
    """
    Main tabu search execution procedure

    :param custList: initial list of customers
    :param matrOfDistances: matrix of distances between all customers
    :param listOfRoutes: initial list of routes to be modified
    :param shapes: shape parameter
    :param scale: scale parameter
    :return: best list of routes and a corresponding objective function
    """

    # to begin I need to specify all the required parameters
    max_iter = 100
    no_impr_iter_max = 30
    max_cand_list_len = 30
    iteration = 0
    no_impr_iter = 0

    # specify a random seed
    random.seed(7)

    # create an empty tabu list.
    # Do I need it to be a dict?

    tabu_list = []
    max_tabu_len = 10

    # create a candidate list
    candidate_list = []

    # set curr_sol and best_sol
    curr_sol = [copy.deepcopy(listOfRoutes), total_obj_fcn_route(listOfRoutes)]
    best_sol = [copy.deepcopy(listOfRoutes), total_obj_fcn_route(listOfRoutes)]

    # start outer main while loop
    while iteration <= max_iter and no_impr_iter <= no_impr_iter_max:
        # print("curr iter ", iteration)
        # print("no impr ", no_impr_iter)

        # create a candidate list
        candidate_list = create_cand_list(custList, curr_sol, max_cand_list_len, matrOfDistances, shapes, scale)

        # at this point I have a full candidate list. Next step is to check for the best elem in the tabu search or
        # gives better result than current best

        # check if best solution in the candidate list is better than current best:

        if candidate_list[0][0][1] < best_sol[1]:

            # if so, need to update best solution, perhaps use deepcopy
            best_sol = [candidate_list[0][0][2], candidate_list[0][0][1]]
            # update current solution
            curr_sol = [candidate_list[0][0][2], candidate_list[0][0][1]]
            # update tabu list -> [(cust1, cust2)]
            tabu_list.append(candidate_list[0][0][0])
            # also need to append reverse of 2 customers
            mirror_move = (candidate_list[0][0][0][1], candidate_list[0][0][0][0])
            tabu_list.append(mirror_move)

            # update tabu list if it is longer than needed
            # tabu list should also take into account possible repetitions
            if len(tabu_list) > max_tabu_len:
                tabu_list = tabu_list[2:]

            # increment iteration count and set no_impr_iter to 0
            # iteration += 1
            no_impr_iter = 0

        # if current best solution in the candidate list is not better than best, following options are possible:
        # 1) it is not in the tabu list -> update current solution, update tabu list, no_impr_iter+=1
        # 2) it is in the tabu list -> find the best candidate not in tabu, update tabu list, no_impr_iter+=1

        # if the best candidate move not in tabu list, this works nicely
        elif candidate_list[0][0][0] not in tabu_list:
            # update current solution
            curr_sol = [candidate_list[0][0][2], candidate_list[0][0][1]]
            # update tabu list
            tabu_list.append(candidate_list[0][0][0])
            # also need to append reverse of 2 customers
            mirror_move = (candidate_list[0][0][0][1], candidate_list[0][0][0][0])
            tabu_list.append(mirror_move)

            if len(tabu_list) > max_tabu_len:
                tabu_list = tabu_list[2:]
            # increase no best solution counter by 1
            no_impr_iter += 1

        # best candidate solution is in the tabu list
        else:
            for elem in candidate_list:
                # elem[0][0] returns a pair of customers

                if elem[0][0] not in tabu_list:
                    curr_sol = [elem[0][2], elem[0][1]]
                    # update tabu list
                    tabu_list.append(elem[0][0])
                    mirror_move = (elem[0][0][1], elem[0][0][0])
                    tabu_list.append(mirror_move)
                    if len(tabu_list) > max_tabu_len:
                        tabu_list = tabu_list[2:]
                    # increase no best solution counter by 1
                    no_impr_iter += 1
                    break

        # empty the candidate list and increase main iteration counter
        iteration += 1

    print("last iter  ", iteration)
    print("no impr iter ", no_impr_iter)

    # return the final list of routes and a corresponding objective function
    return [best_sol, iteration, no_impr_iter]


def whole_route_shift(routePlan, scale, time_shift):
    """

        this function implements a whole route shifting to the right

    :param routePlan: a set of routes to be shifted (footnote: final_ans[0][0])
    :param scale: a scale param of gamma distribution
    :param time_shift: how far we'd like to shift a route.
    :return:
    """

    # I start with the whole set of routes. Step 1 is to properly identify what and for how much I need to change.
    # create a set of shifting values and divide them by the value of scale

    # keep in mind these number are in mins
    shift = time_shift / scale

    best_so_far_obj = 999999
    # now for every route I need to compute the mean of arrival time without taking scale into account
    # similar to the earliness computations

    route_original_shapes = []
    cust_list_ids = []

    obj_after_shift = []
    shift_is = []

    # every route
    for route in routePlan:

        # create an empty list of shapes
        exp_arrival_shape = [0] * len(route.distances)

        # compute all the expected shape of arrivals
        for index, elem in enumerate(exp_arrival_shape):
            if index == 0:
                exp_arrival_shape[index] = route.distances[index] * route.shapes[index]
            else:
                # I need to not just add a previous exp arrival time, but also the time to return
                exp_arrival_shape[index] = exp_arrival_shape[index - 1] + route.distances[index - 1] * route.shapes[
                    index - 1] + route.distances[index] * route.shapes[index]

        # at this point I don't have a depot in my lists
        # add all route shapes to a list
        route_original_shapes.append(exp_arrival_shape[1:])
        # I also need to store customers to get access to che corresp tws
        cust_list_ids.append(route.currentRoute[1:])

    # so I have a set of shapes and scale I need to do shiftings along with corresponding customer id's.
    # I need to take every element (route) in route_original_shapes, add to inner elems the shift and compute
    # the total value of the objective function. keep the best shift and obj function value

    for counter, arrival_shapes in enumerate(route_original_shapes):
        for i in range(0, 13, 1):
            # shift exp arrival times
            arrival_shapes_mod = [x + shift * i for x in arrival_shapes]

            tmp = len(arrival_shapes_mod)

            # compute the obj function values
            earliness = [0] * len(arrival_shapes_mod)
            lateness = [0] * len(arrival_shapes_mod)

            for item in range(0, tmp):
                earliness[item] = expected_earliness(arrival_shapes_mod[item], scale,
                                                     cust_list_ids[counter][item].getEarlyTW())
                lateness[item] = expected_delay(arrival_shapes_mod[item], scale,
                                                cust_list_ids[counter][item].getLateTW())

            # update the best place holder

            tmp_obj = sum(earliness) + sum(lateness)

            # update placeholder for the best obj function value and index of i (i.e. shift)
            if tmp_obj < best_so_far_obj:
                best_so_far_obj = tmp_obj
                best_so_far_i = i

        obj_after_shift.append(best_so_far_obj)
        shift_is.append(best_so_far_i)
        best_so_far_obj = 999

    return obj_after_shift, shift_is, sum(obj_after_shift)


def forward_shifting(routePlan, scale, time_shift):
    # this function implements forward shifting across all customers
    """
    the approach is somewhat similar to the previous one. Instead of shifting all arrival times, take i-th customer and
    check all possible shiftings. If there is a good one, use it and apply the same for all subsequent customers.
    If objective is worse, reverse and move to the next customer. Repeat

    """
    ans = 0

    # I start with the whole set of routes. Step 1 is to properly identify what and for how much I need to change.
    # create a set of shifting values and divide them by the value of scale

    shift_unit = time_shift / scale

    # a set of shifting times (if shifting unit is 5min then a set is up to 1 hour)
    shifting_set = [shift_unit * i for i in range(0, 13)]

    best_so_far_obj = 999999
    # now for every route I need to compute the mean of arrival time without taking scale into account
    # similar to the earliness computations

    route_original_shapes = []
    cust_list_ids = []

    # every route
    for route in routePlan:

        # create an empty list of shapes
        exp_arrival_shape = [0] * len(route.distances)

        # compute all the expected shape of arrivals
        for index, elem in enumerate(exp_arrival_shape):
            if index == 0:
                exp_arrival_shape[index] = route.distances[index] * route.shapes[index]
            else:
                # I need to not just add a previous exp arrival time, but also the time to return
                exp_arrival_shape[index] = exp_arrival_shape[index - 1] + route.distances[index - 1] * route.shapes[
                    index - 1] + route.distances[index] * route.shapes[index]

        # at this point I don't have a depot in my lists
        # add all route shapes to a list
        route_original_shapes.append(exp_arrival_shape[1:])
        # I also need to store customers to get access to che corresp tws
        cust_list_ids.append(route.currentRoute[1:])

        # I have a list of original arrival shapes and a list of customer indices in all routes

        # Now:
        # for every element of original_shapes
        # add shift and compute objective
        # select the best one and modify all the shapes,
        # move to the nex element (possibly use an index)

        by_cust_shift = []
        by_cust_shift_per_route = []
        # for every route
    for counter, arrival_shapes in enumerate(route_original_shapes):
        # for every customer in that route
        for i in range(0, len(arrival_shapes)):
            # add a shift across all possible shifts to an arrival shapes

            ''' possibly if shifting set is not empty '''
            for shift_id, shift in enumerate(shifting_set):

                shape_mod_tmp = arrival_shapes[i:]
                shape_mod = [x + shift for x in shape_mod_tmp]

                # after the shift is added I need to compute objective value

                earliness = [0] * len(shape_mod)
                lateness = [0] * len(shape_mod)

                for item in range(0, len(shape_mod)):
                    earliness[item] = expected_earliness(shape_mod[item], scale,
                                                         cust_list_ids[counter][item + i].getEarlyTW())
                    lateness[item] = expected_delay(shape_mod[item], scale,
                                                    cust_list_ids[counter][item + i].getLateTW())

                tmp_obj = sum(earliness) + sum(lateness)

                # update placeholder for the best obj function value and index of i (i.e. shift)
                if tmp_obj < best_so_far_obj:
                    best_so_far_obj = tmp_obj
                    best_so_far_shift = shift_id

            # after the best shift for current customer is identified, apply that shift to all custs and move
            # to the next customer

            best_so_far_obj = 999999
            by_cust_shift.append(best_so_far_shift)

            ''' this needs to be updated'''
            arrival_shapes_tmp = [x + shifting_set[best_so_far_shift] for x in arrival_shapes]
            arrival_shapes = arrival_shapes_tmp

            # and I need to update a set of possible shifts
            # Now I am not sure I need to do it anymore
            # shifting_set = shifting_set[best_so_far_shift:]

        by_cust_shift_per_route.append(by_cust_shift)
        by_cust_shift = []

        # also need to get a final obj function recomputation after all shifts are complete
        # in order to do this, I need to update all expected arrival times accordingly to shifts for all routes
        # involved

        inc_list_whole = []

    # create a list of increments to be applied
    for i, elem in enumerate(by_cust_shift_per_route):

        inc_list_route = [0] * len(elem)

        for j, item in enumerate(elem):
            if j > 0:
                inc_list_route[j] = item * time_shift + inc_list_route[j - 1]
            else:
                inc_list_route[j] = item * time_shift

        inc_list_whole.append(inc_list_route)

        # modify route arrival times

    exp_arrival_shape_after_shifts_per_routes = []

    for i, elem in enumerate(route_original_shapes):
        exp_arrival_shape_after_shifts = [0] * len(elem)
        for j, item in enumerate(elem):
            exp_arrival_shape_after_shifts[j] = item + inc_list_whole[i][j]
        exp_arrival_shape_after_shifts_per_routes.append(exp_arrival_shape_after_shifts)

    # recompute final obj value
    total_obj = 0
    obj_e = 0
    obj_l = 0

    for i, item in enumerate(exp_arrival_shape_after_shifts_per_routes):
        # elem is just an exp arrival time to a cust

        earliness = [0] * len(item)
        lateness = [0] * len(item)

        for j, elem in enumerate(item):
            earliness[j] = expected_earliness(elem, scale,
                                              cust_list_ids[i][j].getEarlyTW())
            lateness[j] = expected_delay(elem, scale,
                                         cust_list_ids[i][j].getLateTW())

        obj = sum(earliness) + sum(lateness)
        obj_e += sum(earliness)
        obj_l += sum(lateness)
        total_obj += obj

    return by_cust_shift_per_route, total_obj, obj_e / total_obj, obj_l / total_obj


# extra initialization alg
"""
    # After having a list of customers and routes I need to create initial set of routes filed with customers.
    # Step 1: take empty routes. For all customers check all insertion positions based on the measures and insert
    # those customers 1 by 1.

    # need to replace this while by for indexed over the number of customers in the customer list

    for i in range(0, len(custList)):
        print(len(custList))
        obj_fun_change = float("inf")

        # for every customer and route
        for customer in custList:
            for route in routePlan:
                # for every position
                for position, elem in enumerate(route.currentRoute, 1):
                    # compute measures before and after the customer is inserted
                    prev_lateness = route.total_lateness()
                    prev_earliness = route.total_earliness()
                    route.insert_customer(position, customer, distances, shapePar, scalePar)
                    curr_lateness = route.total_lateness()
                    curr_earliness = route.total_earliness()
                    # change in the earl\laten
                    change_in_measures = curr_earliness - prev_earliness + curr_lateness - prev_lateness

                    # update the insertion location if needed
                    if change_in_measures < obj_fun_change:
                        cust_to_ins = customer
                        route_to_ins = route.id
                        pos_to_ins = position
                        # need to update upper bound on the measure to make sure it updates when needed, not always
                        obj_fun_change = change_in_measures
                    # once everything is computed and compared, restore the route
                    route.remove_customer(position, distances, shapePar, scalePar)

        # after all customers and positions are checked finally insert a customer into a route
        routePlan[route_to_ins].insert_customer(pos_to_ins, cust_to_ins, distances, shapePar, scalePar)
        # and remove a customer from a list of initial customers
        custList.remove(cust_to_ins)
"""

"""
tw shift


# [xLow, yLow, xUp, yUp]
zoneCoords = [.3, .6, 1.7, 1.3]

custList = []
with open('9.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        custList.append(Customer(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

for elem in custList:
    if zoneCoords[0] < elem.xCoord < zoneCoords[2] and zoneCoords[1] < elem.yCoord < zoneCoords[3]:
        if elem.lateTW < 8:
            elem.lateTW += 1
        else:
            elem.earlyTW -= 1

# now all the tws are modified. The next step is to override the csv file

with open('9.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    for elem in custList:
        writer.writerow([elem.id, elem.xCoord, elem.yCoord, elem.earlyTW, elem.lateTW])

csv_file.close()

"""
