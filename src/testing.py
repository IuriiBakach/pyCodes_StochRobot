import csv
import numpy as np

from classes_needed import *

# def main():
# set initial parameters
robotSpeed = 3

# Create an initial matrix for shape and scale values for different zones and times
# col = zone, row = hours
los_matrix = np.zeros((1, 2), dtype='f,f').tolist()

# _____parameters of the gamma distribution; order -> (shape, scale) outer zone, (shape, scale) inner zone
los_matrix[0] = [(1, 1), (4, 1)]

print("Outer zone shape is {} and inner zone shape is {}".format(los_matrix[0][0][0], los_matrix[0][1][0]))

# create initial data: customers, depots, routes.

# how about create a list with a set of csv files and then run the alg over all of them
csv_list = ['new_data_c_20_p_65_3.csv']  # , '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv']

for item in csv_list:
    # read in customers
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

    # distance is a np.array of best paths distances both out and in the zone
    # shapes are multipliers (shape coeffs) to get expected arrival time based on distances. They are also based on
    # robot speed. i.e. they are divided by it

    distances, shapes, path_indices, best_paths = dist_matr_trim(distances_raw, los_matrix, custList)
    '''
    # get total distances to customers in km as np array

    best_paths_distance_combined = []
    for index, elem in enumerate(best_paths):
        if elem == 0:
            best_paths_distance_combined.append(elem)
        else:
            best_paths_distance_combined.append(elem[0] + elem[1])

    # at this point all distances are one-way distances. They are needed to be multiplied by 2 to get the full picture
    best_paths_distance_combined = np.asarray(best_paths_distance_combined)

    total_exp_distance = sum(best_paths_distance_combined) * 2

    # add all customers
    routePlan[0].insert_customer_v_2(1, custList[4], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[1], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[9], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[13], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[18], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[15], distances, shapes, los_matrix[0][0][1])

    routePlan[1].insert_customer_v_2(1, custList[19], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[11], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[10], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[17], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[6], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[8], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[2], distances, shapes, los_matrix[0][0][1])

    routePlan[2].insert_customer_v_2(1, custList[12], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[16], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[14], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[7], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[5], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[3], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[0], distances, shapes, los_matrix[0][0][1])

 

    routePlan[0].insert_customer_v_2(1, custList[19], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[4], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[1], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[14], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[9], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[17], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[0], distances, shapes, los_matrix[0][0][1])
    routePlan[0].insert_customer_v_2(1, custList[8], distances, shapes, los_matrix[0][0][1])

    routePlan[1].insert_customer_v_2(1, custList[7], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[13], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[6], distances, shapes, los_matrix[0][0][1])
    routePlan[1].insert_customer_v_2(1, custList[2], distances, shapes, los_matrix[0][0][1])

    routePlan[2].insert_customer_v_2(1, custList[12], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[16], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[11], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[10], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[5], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[3], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[18], distances, shapes, los_matrix[0][0][1])
    routePlan[2].insert_customer_v_2(1, custList[15], distances, shapes, los_matrix[0][0][1])

    # routePlan[0].remove_customer_v_2(2, los_matrix[0][0][1])

    # insertion and removal work. What next?
    # next is tabu search. namely 1-shift and exchange
    # run tabu search

    # final_ans = tabu_search(custList_tabu, distances, routePlan, shapes, los_matrix[0][0][1])

    after_shift_obj, after_shift_ids, total_after_shift_obj = whole_route_shift(routePlan, los_matrix[0][0][1],
                                                                                5)

    # final_ans[0][0]

    by_cust_shift_per_route, obj_value_after_fwd_shift, percent_early, percent_late = forward_shifting(routePlan,
                                                                                                       los_matrix[0][0][
                                                                                                           1], 5)

    # print(final_ans[0][0][0].total_lateness() + final_ans[0][0][1].total_lateness() + final_ans[0][0][
    #    0].total_earliness() + final_ans[0][0][
    #          1].total_earliness() + final_ans[0][0][2].total_lateness() + final_ans[0][0][2].total_earliness())

    print(total_after_shift_obj)

    print(obj_value_after_fwd_shift)

    print(percent_early)
    print(percent_late)
'''
