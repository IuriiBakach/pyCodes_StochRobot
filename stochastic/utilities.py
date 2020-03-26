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
        if cust.getYCoord() > inner_zone_coords[1] or 0 < cust.getXCoord() < inner_zone_coords[0] and \
                inner_zone_coords[2] < cust.getXCoord() < 2:
            # compute manhattan distance
            case_distance = (abs(cust.getXCoord() - depot_loc[0]) + abs(cust.getYCoord() - depot_loc[1]), 0)
            # modify the whole row so that all the distances are the same for 4 possible routes
            for i in range(0, 4):
                tmp.append(case_distance)

                # check distance matrix indices
            distance_matrix[index] = tmp
            tmp = []

        # if if customer falls into case 3, i.e. is in the lower part of the graph directly below the zone.
        # in this case we have 2 possible routes:
        # 1) shortest through the zone
        # 2) shortest avoiding the zone
        # other 2 routes should be filled with something (perhaps inf?)
        if inner_zone_coords[0] < cust.getXCoord() < inner_zone_coords[2] and \
                0 < cust.getYCoord() < inner_zone_coords[1]:
            # 1) shortest through the zone, (blocks outside inner zone, blocks inside inner zone)
            case1_distance = (abs(cust.getXCoord() - depot_loc[0]) + abs(cust.getYCoord() - depot_loc[1]) -
                              (inner_zone_coords[3] + inner_zone_coords[1]),
                              inner_zone_coords[3] + inner_zone_coords[1])
            # 2) shortest avoiding the zone
            case2_distance_a = abs(inner_zone_coords[0] - depot_loc[0]) + abs(inner_zone_coords[1] - depot_loc[1]) + \
                               abs(inner_zone_coords[0] - cust.getXCoord()) + abs(
                inner_zone_coords[1] - cust.getYCoord())

            case2_distance_b = abs(inner_zone_coords[2] - depot_loc[0]) + abs(
                inner_zone_coords[1] - depot_loc[1]) + abs(inner_zone_coords[2] - cust.getXCoord()) + abs(
                inner_zone_coords[1] - cust.getYCoord())

            # fill the proper row that corresponds to a customer
            case2_distance = (min(case2_distance_a, case2_distance_b), 0)
            tmp = [case1_distance, case2_distance, (1000, 1000), (1000, 1000)]

            # check distance matrix indices
            distance_matrix[index] = tmp
            tmp = []

        # if customer falls into case 2, i.e. has coord in the zone, then 4 different paths are possible.
        else:
            # step 1: find 4 entry points: direct distances from a customer to a edges and these are inner distances
            # points: 1-left, 2-down, 3-up, 4-right
            # a point contain coords and number of blocks inside a zone

            entry_pt_1 = [(inner_zone_coords[0], cust.getYCoord()), abs(inner_zone_coords[0] - cust.getXCoord())]
            entry_pt_2 = [(cust.getXCoord(), inner_zone_coords[1]), abs(inner_zone_coords[1] - cust.getYCoord())]
            entry_pt_3 = [(cust.getXCoord(), inner_zone_coords[2]), abs(inner_zone_coords[2] - cust.getYCoord())]
            entry_pt_4 = [(inner_zone_coords[3], cust.getYCoord()), abs(inner_zone_coords[3] - cust.getXCoord())]

            # step 2: find the distance to all entry points avoiding inner zone
            # to points 1, 3, 4 it's easy; to point 2 need to use same approach as for the case 3 -> min(A, B)

            case2_distance_1 = (abs(depot_loc[0] - entry_pt_1[0][0]) +
                                abs(depot_loc[1] - entry_pt_1[0][1]), entry_pt_1[1])
            case2_distance_3 = (abs(depot_loc[0] - entry_pt_3[0][0]) +
                                abs(depot_loc[1] - entry_pt_3[0][1]), entry_pt_3[1])
            case2_distance_4 = (abs(depot_loc[0] - entry_pt_4[0][0]) +
                                abs(depot_loc[1] - entry_pt_4[0][1]), entry_pt_4[1])

            case2_distance_2 =

    return distance_matrix


def expected_delay(shape, scale, uppertw):
    """
        This function computes expected delay of the arrival of robot r to the customer c taking into account
    all previously visited customers

    input parameters:
    shape- shape parameter obtained as distance (travel time) traveled till this customer,
    corresponds to _jv in the paper
    scale- scale parameter (fixed for all customers in the specific run)
    upperTW- late time window for a customer
    :return: expected delay
    """

    # ______ create required gamma distributions 3 modify initial creation of the distributions
    gd1 = gamma(shape)
    gd2 = gamma(shape + 1)
    ans = shape * scale * (1 - gd2.cdf(uppertw)) - uppertw * (1 - gd1.cdf(uppertw))
    '''
    if ans < 0:
        return 0.0
    else:
    '''
    return ans


def expected_earliness(shape, scale, lowertw):
    """
        This function computes expected earliness of the arrival of robot r to the customer c taking into account
    all previously visited customers

    input parameters:
        shape- shape parameter obtained as distance (travel time) traveled till this customer
        scale- scale parameter (fixed for all customers in the specific run)
        upperTW- late time window for a customer

    :return: expected earliness
    """

    gd1 = gamma(shape)
    gd2 = gamma(shape + 1)

    ans = lowertw * gd1.cdf(lowertw) - shape * scale * gd2.cdf(lowertw)
    '''
    if ans < 0:
        return 0.0
    else:
    '''
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


def create_cand_list(customerList, currentSolutionRoutes, max_cand_list_len, distances, shape, scale):
    """
    Create a candidate list of the len = max_cand_list_len for the tabu search

    :param customerList: initial customer list
    :param currentSolutionRoutes: current solution
    :param max_cand_list_len: how many candidates to create
    :param distances: matrix of distacne between customers
    :param shape: shape
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

            operation = one_shift(customerList[r_cust_one - 1], customerList[r_cust_two - 1], currentSolutionRoutes[0],
                                  distances, shape, scale)

        else:
            # perform exchange operation
            operation = exchange(customerList[r_cust_one - 1], customerList[r_cust_two - 1], currentSolutionRoutes[0],
                                 distances, shape, scale)

        # I need to add to a candidate list neighborhood and operation id (either 1 or 2)
        candidate_list.append([operation, r_operation])
    # Sorted candidate list based on the changed objective function
    # now I can take first element of whatever I need
    candidate_list.sort(key=lambda x: x[0][1])

    return candidate_list


def tabu_search(custList, matrOfDistances, listOfRoutes, shape, scale):
    """
    Main tabu search execution procedure

    :param custList: initial list of customers
    :param matrOfDistances: matrix of distances between all customers
    :param listOfRoutes: initial list of routes to be modified
    :param shape: shape parameter
    :param scale: scale parameter
    :return: best list of routes and a corresponding objective function
    """

    # to begin I need to specify all the required parameters
    max_iter = 100
    no_impr_iter_max = 20
    max_cand_list_len = 50
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
        print("curr iter ", iteration)
        print("no impr ", no_impr_iter)

        # create a candidate list
        candidate_list = create_cand_list(custList, curr_sol, max_cand_list_len, matrOfDistances, shape, scale)

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

    # return the final list of routes and a corresponding objective function
    return [best_sol, iteration, no_impr_iter]
