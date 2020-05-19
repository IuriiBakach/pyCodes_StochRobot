import csv
import numpy as np

from classes_needed import *


def main():
    # set initial parameters
    robotSpeed = 3
    maxTravelDist = 6

    # Create an initial matrix for shape and scale values for different zones and times
    # col = zone, row = hours
    los_matrix = np.zeros((1, 2), dtype='f,f').tolist()

    # _____parameters of the gamma distribution; order -> shape outer zone, shape inner zone
    los_matrix[0] = [(1, 1), (4, 1)]
    # these are just stub values, potentially to update later

    shapePar = 1
    scalePar = 1
    # create initial data: customers, depots, routes.

    # create depot with specified number of robots. Perhaps I don't really need a class-> dict would work

    depot = Depot(0, 0, 0)
    depot.setNumberOfRobots(3)

    # read in customers

    custList = []
    with open('10.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            custList.append(Customer(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    # create a corresponding RoutePlan based on the number of robots in the depot

    routePlan = []

    # create a copy of customers for the tabu search
    custList_tabu = copy.deepcopy(custList)

    # create a list of routes with the number of empty routes corresponding to the number of robots in the depot

    for elem in range(depot.getNumberRobots()):
        routePlan.append(Route(elem))

    depotCoords = [1.2, 1.7]
    zoneCoords = [.3, .6, 1.7, 1.3]

    # create a matrix of distances from depot(s) to customers

    distances_raw = all_distances(depotCoords, custList, zoneCoords)
    distances, path_indices, best_paths = dist_matr_trim(distances_raw, los_matrix, custList)

    # print(distances_raw)
    # print(distances)
    # print(path_indices)
    # print(best_paths)

    # get total distances to customers in km as np array

    best_paths_distance_combined = []
    for index, elem in enumerate(best_paths):
        if elem == 0:
            best_paths_distance_combined.append(elem)
        else:
            best_paths_distance_combined.append(elem[0] + elem[1])

    # at this point all distances are one-way distances. They are needed to be multiplied by 2 to get the full picture
    best_paths_distance_combined = np.asarray(best_paths_distance_combined)

    # potentially there is a need to further diversify the customer and paths to see what customer is in what zone

    # print the output data:
    # total distance
    # total expected travel time
    # paths selected

    # total_exp_travel_time = 0
    total_exp_distance = 0

    # stub = distances[0] - 1 / 30
    # stub[0] = 0

    # total_exp_travel_time = sum(stub) * 2
    total_exp_distance = sum(best_paths_distance_combined) * 2

    '''
    distances = np.zeros((1, len(custList) + 1))
    
    # ________ this returns a vector of distances form a depot to every customer in hours with robots speed;
    # not round trip. robot speed to be removed when corresponding alphas are taken into account
    for i in range(0, len(custList)):
        distances[0][i + 1] = (abs(custList[i].xCoord - depot.xCoord) + abs(custList[i].yCoord - depot.yCoord))/robotSpeed
    '''

    '''
    print(distances)
    # seems lite that's it for the init phase
    
    # now I need to fill initial routes with customers
    print(routePlan[0])
    
    routePlan[0].insert_customer(2, custList[3], distances, shapePar, scalePar)
    print(routePlan[0])
    
    routePlan[0].insert_customer(2, custList[0], distances, shapePar, scalePar)
    print(routePlan[0])
    
    '''
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

    routePlan[0].insert_customer(1, custList[0], distances, shapePar, scalePar)
    routePlan[0].insert_customer(2, custList[1], distances, shapePar, scalePar)
    routePlan[0].insert_customer(2, custList[2], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[3], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[4], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[5], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[6], distances, shapePar, scalePar)

    routePlan[2].insert_customer(1, custList[7], distances, shapePar, scalePar)
    routePlan[2].insert_customer(1, custList[8], distances, shapePar, scalePar)
    routePlan[2].insert_customer(1, custList[9], distances, shapePar, scalePar)
    routePlan[2].insert_customer(1, custList[10], distances, shapePar, scalePar)
    routePlan[2].insert_customer(1, custList[11], distances, shapePar, scalePar)
    routePlan[2].insert_customer(1, custList[12], distances, shapePar, scalePar)

    routePlan[1].insert_customer(1, custList[13], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[14], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[15], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[16], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[17], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[18], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[19], distances, shapePar, scalePar)

    # need to run this tabu search
    final_ans = tabu_search(custList_tabu, distances, routePlan, shapePar, scalePar)

    # results outputs
    best_paths = best_paths[1:]

    # consider total travelled inner and outer distances
    total_exp_distance_inner = 0
    total_exp_distance_outer = 0

    for elem in best_paths:
        total_exp_distance_inner += elem[1]
        total_exp_distance_outer += elem[0]

    print("Traveled distance INside the zone is {} ".format(total_exp_distance_inner))
    print("Traveled distance OUTside the zone is {} ".format(total_exp_distance_outer))
    print("Total distance traveled is {} ".format(total_exp_distance))

    # need to compute expected travel times in inner/outer zones

    total_exp_travel_time_inner = 0
    total_exp_travel_time_outer = 0

    for elem in best_paths:
        total_exp_travel_time_inner += elem[1] * los_matrix[0][1][0]
        total_exp_travel_time_outer += elem[0] * los_matrix[0][0][0]

    total_exp_travel_time_inner = 2 * total_exp_travel_time_inner / robotSpeed
    total_exp_travel_time_outer = 2 * total_exp_travel_time_outer / robotSpeed

    print("Expected inner travel time is {:.6} ".format(total_exp_travel_time_inner))
    print("Expected outer travel time is {:.6} ".format(total_exp_travel_time_outer))

    total_exp_travel_time = 0

    # for each elem in the best path compute corresponding total expected travel times
    for elem in best_paths:
        total_exp_travel_time += elem[0] * los_matrix[0][0][0] + elem[1] * los_matrix[0][1][0]

    # to get a total expected travel time I need to multiply it by 2
    total_exp_travel_time = 2 * total_exp_travel_time / robotSpeed

    print("Total expected travel time is {:.6} ".format(total_exp_travel_time))
    print("Paths selected are {} ".format(path_indices))

    print("Final set of routes: ", final_ans[0][0])

    t_earl = 0
    t_lateness = 0

    # Total earliness by routes
    for elem in final_ans[0][0]:
        t_earl += elem.total_earliness()

    print('Percentage of earliness is {:.4}'.format(t_earl / final_ans[0][1]))

    # Total lateness by routes
    for elem in final_ans[0][0]:
        t_lateness += elem.total_lateness()

    print('Percentage of lateness is {:.4}'.format(t_lateness / final_ans[0][1]))

    print("Total obj function is {:.6} ".format(final_ans[0][1]))


if __name__ == "__main__":
    main()
