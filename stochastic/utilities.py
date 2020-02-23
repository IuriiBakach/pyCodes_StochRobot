from scipy.stats import gamma
from classes_needed import *
import copy
import random
from random import randint


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

    # ______ create required gamma distributions
    gd1 = gamma(shape, 1 / scale)
    gd2 = gamma(shape + 1, 1 / scale)

    return shape * scale * (1 - gd2.cdf(uppertw)) - uppertw * (1 - gd1.cdf(uppertw))


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

    gd1 = gamma(shape, 1 / scale)
    gd2 = gamma(shape + 1, 1 / scale)

    return lowertw * gd1.cdf(lowertw) - shape * scale * gd2.cdf(lowertw)


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


def m13(route1, route2):
    """
    This function computes increase in the linear combination of earliness and lateness between before and after
    a customer is inserted into a certain position in the route

    additional multipliers could be used to emphasize on earliness/lateness

    :param route1: route before the customer is inserted
    :param route2: route after the customer is inserted
    :return: the change in the objective function
    """

    return (route2.total_earliness() - route1.total_earliness()) + (route2.total_lateness() - route1.total_lateness())


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

    route_plan_copy[loc_cust_one[0]].remove_customer(position_cust_one, distMatr, shape, scale)
    route_plan_copy[loc_cust_two[0]].remove_customer(position_cust_two, distMatr, shape, scale)

    # check indexes so that everything works nicely
    route_plan_copy[loc_cust_one[0]].insert_customer(position_cust_one, cust2, distMatr, shape, scale)
    route_plan_copy[loc_cust_two[0]].insert_customer(position_cust_two, cust1, distMatr, shape, scale)

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


def tabu_search(custList, matrOfDistances, listOfRoutes, shape, scale):
    """
    Main tabu search execution procedure

    :param custList:
    :param matrOfDistances:
    :param listOfRoutes:
    :param shape:
    :param scale:
    :return:
    """

    # to begin I need to specify all the required parameters
    max_iter = 100
    no_impr_iter_max = 10
    max_cand_list_len = 7
    iteration = 0
    no_impr_iter = 0

    # specify a random seed
    random.seed(7)

    # create an empty tabu list.
    # Do I need it to be a dict?

    tabu_list = []
    max_tabu_len = 5

    # create a candidate list
    candidate_list = []

    # set curr_sol and best_sol
    curr_sol = [copy.deepcopy(listOfRoutes), total_obj_fcn_route(listOfRoutes)]
    best_sol = [copy.deepcopy(listOfRoutes), total_obj_fcn_route(listOfRoutes)]

    # start outer main while loop
    while iteration <= max_iter and no_impr_iter <= no_impr_iter_max:
        while len(candidate_list) <= max_cand_list_len:
            # select vertices randomly
            # here I get the INDEX of the customers I'd like to use in the swap
            random_sample = random.sample(range(1, len(custList) - 1), 2)
            r_cust_one = random_sample[0]
            r_cust_two = random_sample[1]

            # once I selected these indices, I need to randomly chose an operation
            r_operation = randint(1, 2)
            if r_operation == 1:
                # I need to perform a one_shift operation
                # operation has [(cust1, cust2), obj_fcn, modifiedRoute]
                # listOfRoutes is the current solution

                # also need to check if this works after deepcopy!!!

                operation = one_shift(custList[r_cust_one], custList[r_cust_two], listOfRoutes, matrOfDistances, shape,
                                      scale)

            else:
                # perform exchange operation
                operation = exchange(custList[r_cust_one], custList[r_cust_two], listOfRoutes, matrOfDistances, shape,
                                     scale)

            # I need to add to a candidate list neighborhood and operation id (either 1 or 2)
            candidate_list.append([operation, r_operation])
        # Sorted candidate list based on the changed objective function
        # now I can take first element of whatever I need
        candidate_list.sort(key=lambda x: x[0][1])

    # at this point I have a full candidate list. Next step is to check for the best elem in the tabu search or
    # gives better result than current best

    # check if best solution in the candidate list is better than current best:
        if candidate_list[0][0][1] < best_sol[1]:
            # if so, need to update best solution, perhaps use deepcopy
            best_sol = [candidate_list[0][0][2], candidate_list[0][0][1]]
            # update current solution
            curr_sol = [candidate_list[0][0][2], candidate_list[0][0][1]]
            # update tabu list -> [(cust1, cust2), move]
            tabu_list.append([candidate_list[0][0][0], candidate_list[0][1]])

            # update tabu list if it is longer than needed
            # tabu list should also take into account possible repetitions
            if len(tabu_list) > max_tabu_len:
                tabu_list.pop(0)

            # increment iteration count and set no_impr_iter to 0
            iteration +=1
            no_impr_iter = 0
        else:
            # check if the best move is in the tabu list
            # if not, update current solution, increase no_impr_iter by 1 and iteration by 1
            # if yes, find the best one not in tabu, and increase all counters
            # do not forget update tabu list here too
            do_stuff_here

    # empty the candidate list
    candidate_list = []

    return 0