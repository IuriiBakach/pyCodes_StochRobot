import csv
import itertools
import numpy as np

from classes_needed import *

# set initial parameters
robotSpeed = 3
maxTravelDist = 6

# Create an initial matrix for shape and scale values for different zones and times
# col = zone, row = hours
los_matrix = np.zeros((1, 2), dtype='f,f').tolist()

# _____parameters of the gamma distribution
los_matrix[0] = [(1, 1), (2, 1)]
# these are just stub values, potentially to update later

shapePar = 1
scalePar = 1
# create initial data: customers, depots, routes.

# create depot with specified number of robots. Perhaps I don't really need a class-> dict would work

depot = Depot(0, 0, 0)
depot.setNumberOfRobots(1)

# create customers. This should be read in form the .csv file but it's ok for now
# 480 correspond to 8am. 1020 to 5pm
# now I think all time windows should be scaled to 8 am -> 0

custList = []
with open('customersTest.csv', 'r') as file:
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
distances, path_indices = dist_matr_trim(distances_raw, los_matrix, custList)

print(type(distances))

# for elem in distances:
#    for elem1 in elem:
#        print(elem1)

# need to run this tabu search

# dists = [[0], [0.56666667], [.8, .7], [1.46666667, .7, .6, 0.53333333], [.86666667, 1.1, .46666667, 1],
#         [0.76666667, .8, 0.76666667, 1.3], [0.83333333, 0.86666667], [0.1], [1.16666667, .9, .4, 0.76666667]]

dists = np.asarray([[0], [0.56666667], [.8, .7], [1.46666667, .7, .6, 0.53333333], [.86666667, 1.1, .46666667, 1],
                    [0.76666667, .8, 0.76666667, 1.3], [0.83333333, 0.86666667], [0.1]])

print(type(dists))

dist_combs = np.asarray(list(itertools.product(*dists)))

stb = [0, 1, 2, 3, 4, 5, 6]
cust_id_combs = np.asarray(list(itertools.permutations(stb)))

# to do: 1) create all possible combinations of customers and paths
# for each customer combination run all the paths and save one with the best obj func value
# report only the best path and a path selection

best_obj = 1000
best_earl = 1000
best_latte = 1000

best_overall_obj = 1000
best_overall_earl = 1000
best_overall_latte = 1000
best_overall_paths = []
best_overall_plan = 0

# for every set of custs
for indexing_custs, cust_elem in enumerate(cust_id_combs):
    # for every set of paths
    for indexing_paths, path_elem in enumerate(dist_combs):
        distances[0] = path_elem
        # add all custs to the route
        routePlan[0].insert_customer(1, custList[cust_elem[0]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(2, custList[cust_elem[1]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(3, custList[cust_elem[2]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(4, custList[cust_elem[3]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(5, custList[cust_elem[4]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(6, custList[cust_elem[5]], distances, shapePar, scalePar)
        routePlan[0].insert_customer(7, custList[cust_elem[6]], distances, shapePar, scalePar)
        # routePlan[0].insert_customer(1, custList[cust_elem[7]], distances, shapePar, scalePar)

        # compute objective function value

        curr_earl = routePlan[0].total_earliness()
        curr_latte = routePlan[0].total_lateness()
        curr_obj = curr_earl + curr_latte

        # update best solution if needed
        if curr_obj < best_obj:
            best_obj = curr_obj
            best_earl = curr_earl
            best_latte = curr_latte
            best_paths = indexing_paths
            curr_plan = routePlan[0]

        routePlan = []
        for elem1 in range(depot.getNumberRobots()):
            routePlan.append(Route(elem1))

        # after all paths for a given route are checked, update best route so far

    if best_obj < best_overall_obj:
        best_overall_obj = best_obj
        best_overall_earl = best_earl
        best_overall_latte = best_latte
        best_overall_paths = best_paths
        best_overall_plan = indexing_custs

    print(indexing_custs)

    best_obj = 1000
    best_earl = 1000
    best_latte = 1000

print("earliness: {}, lateness: {}, obj value: {}, path_set: {}, routePlan: {}".format(best_overall_earl,
                                                                                       best_overall_latte,
                                                                                       best_overall_obj,
                                                                                       best_overall_paths,
                                                                                       best_overall_plan))

'''
for indexing, elem in enumerate(combs):
    # print(distances_raw)
    # print(distances)
    distances[0] = elem

    print(indexing)
    print(distances)

    routePlan[0].insert_customer(1, custList[2], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[3], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[1], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[0], distances, shapePar, scalePar)

    print(routePlan[0].currentRoute)

    curr_earl = routePlan[0].total_earliness()
    curr_latte = routePlan[0].total_lateness()
    curr_obj = curr_earl + curr_latte

    if curr_obj < best_obj:
        best_obj = curr_obj
        best_earl = curr_earl
        best_latte = curr_latte
        best_paths = indexing
        curr_plan = routePlan[0]

    # print(routePlan[0])
    # print(earl)
    # print(latte)
    # print(earl + latte)
    routePlan = []
    for elem1 in range(depot.getNumberRobots()):
        routePlan.append(Route(elem1))
    # print(indexing)

# final_ans = tabu_search(custList_tabu, distances, routePlan, shapePar, scalePar)

# print(final_ans[0])
'''
# print("earliness: {}, lateness: {}, obj value: {}, path_set: {}, routePlan: {}".format(best_earl, best_latte, best_obj,
#                                                                                       best_paths, curr_plan))
