import csv
import numpy as np

from classes_needed import *


def main():
    # set initial parameters
    robotSpeed = 3

    # Create an initial matrix for shape and scale values for different zones and times
    # col = zone, row = hours
    los_matrix = np.zeros((1, 2), dtype='f,f').tolist()

    # _____parameters of the gamma distribution; order -> shape outer zone, shape inner zone
    los_matrix[0] = [(1, 1), (4, 1)]

    print("Outer zone shape is {} and inner zone shape is {}".format(los_matrix[0][0][0], los_matrix[0][1][0]))

    idex = 1

    # how about create a list with a set of csv files and then run the alg over all of them
    # csv_list = ['1.csv' , '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv']
    csv_list = ['new_data_c_20_p_35_1.csv', 'new_data_c_20_p_35_2.csv', 'new_data_c_20_p_35_3.csv',
                'new_data_c_20_p_35_4.csv', 'new_data_c_20_p_35_5.csv', 'new_data_c_20_p_35_6.csv',
                'new_data_c_20_p_35_7.csv',
                'new_data_c_20_p_35_8.csv', 'new_data_c_20_p_35_9.csv', 'new_data_c_20_p_35_10.csv']

    for item in csv_list:
        # read in customers
        print('new_data_c_50_p_35' + '_' + str(idex))
        idex += 1
        custList = []
        with open(item, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                custList.append(Customer(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

        # create depot with specified number of robots. Perhaps I don't really need a class-> dict would work

        depot = Depot(0, 0, 0)
        depot.setNumberOfRobots(3)

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

        # as of now, distances contain (distance, shape coeff)
        distances, shapes, path_indices, best_paths = dist_matr_trim(distances_raw, los_matrix, custList)

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

        # at this point all distances are one-way distances. They are needed to be multiplied by 2 to get the full
        # picture
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

        # initialization alg

        # After having a list of customers and routes I need to create initial set of routes filed with customers.
        # Step 1: take empty routes. For all customers check all insertion positions based on the measures and insert
        # those customers 1 by 1.

        for i in range(0, len(custList)):
            # print(len(custList))
            obj_fun_change = float("inf")

            # for every customer and route
            for customer in custList:
                for route in routePlan:
                    # for every position
                    for position, elem in enumerate(route.currentRoute, 1):
                        # compute measures before and after the customer is inserted
                        prev_lateness = route.total_lateness()
                        prev_earliness = route.total_earliness()
                        route.insert_customer_v_2(position, customer, distances, shapes, los_matrix[0][0][1])
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
                        route.remove_customer_v_2(position, los_matrix[0][0][1])

            # after all customers and positions are checked finally insert a customer into a route
            routePlan[route_to_ins].insert_customer_v_2(pos_to_ins, cust_to_ins, distances, shapes, los_matrix[0][0][1])
            # and remove a customer from a list of initial customers
            custList.remove(cust_to_ins)

        # need to run this tabu search
        final_ans = tabu_search(custList_tabu, distances, routePlan, shapes, los_matrix[0][0][1])

        # results outputs
        best_paths = best_paths[1:]

        # consider total travelled inner and outer distances
        total_exp_distance_inner = 0
        total_exp_distance_outer = 0

        for elem in best_paths:
            total_exp_distance_inner += elem[1]
            total_exp_distance_outer += elem[0]

        print("Traveled distance INside the zone is {} ".format(2 * total_exp_distance_inner))
        print("Traveled distance OUTside the zone is {} ".format(2 * total_exp_distance_outer))
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

        '''last parameter for all shiftings is waiting time. Currently in mins'''

        # evaluate shiftings
        after_shift_obj, after_shift_ids, total_after_shift_obj = whole_route_shift(final_ans[0][0],
                                                                                    los_matrix[0][0][1],
                                                                                    5)
        print("Objective function after a whole route shift {:.6} ".format(total_after_shift_obj))

        by_cust_shift_per_route, obj_value_after_fwd_shift, percent_early, percent_late = forward_shifting(
            final_ans[0][0], los_matrix[0][0][1], 5)

        print("Objective function after a forward shift shift {:.6} ".format(obj_value_after_fwd_shift))

        print('Percentage of earliness is {:.4}'.format(percent_early))
        print('Percentage of lateness is {:.4}'.format(percent_late))

        print("Shifts used: ", by_cust_shift_per_route)


if __name__ == "__main__":
    main()
