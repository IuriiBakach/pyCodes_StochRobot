import csv

import numpy as np

from classes_needed import *

# set initial parameters
robotSpeed = 3
maxTravelDist = 6

# Create an initial matrix for shape and scale values for different zones and times
# col = zone, row = hours
los_matrix = np.zeros((1, 2), dtype='f,f').tolist()
los_matrix[0] = [(1, 1), (1, 1)]
# fill it with appropriate values here


# _____parameters of the gamma distribution
shapePar = 1
scalePar = 1
# create initial data: customers, depots, routes.

# create depot with specified number of robots. Perhaps I don't really need a class-> dict would work

depot = Depot(0, 0, 0)
depot.setNumberOfRobots(3)

# create customers. This should be read in form the .csv file but it's ok for now
# 480 correspond to 8am. 1020 to 5pm
# now I think all time windows should be scaled to 8 am -> 0

custList = []
with open('customers.csv', 'r') as file:
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
print(distances_raw)
print(distances)
print(path_indices)
# for elem in distances:
#    for elem1 in elem:
#        print(elem1)

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


routePlan[0].insert_customer(1, custList[0], distances, shapePar, scalePar)
print(routePlan[0])

routePlan[0].insert_customer(2, custList[1], distances, shapePar, scalePar)
print(routePlan[0])

routePlan[0].remove_customer(2, distances, shapePar, scalePar)
print(routePlan[0])

routePlan[0].insert_customer(1, custList[2], distances, shapePar, scalePar)
print(routePlan[0])

routePlan[0].remove_customer(1, distances, shapePar, scalePar)
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
'''

routePlan[0].insert_customer(1, custList[3], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[19], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[4], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[15], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[16], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[8], distances, shapePar, scalePar)

routePlan[1].insert_customer(1, custList[2], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[11], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[9], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[0], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[12], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[18], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[17], distances, shapePar, scalePar)

routePlan[2].insert_customer(1, custList[7], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[1], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[14], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[6], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[5], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[10], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[13], distances, shapePar, scalePar)

# need to run this tabu search

# earl = routePlan[0].total_earliness() + routePlan[1].total_earliness() + routePlan[2].total_earliness()
# latte = routePlan[0].total_lateness() + routePlan[1].total_lateness() + routePlan[2].total_lateness()

# print(earl + latte)
final_ans = tabu_search(custList_tabu, distances, routePlan, shapePar, scalePar)

print(final_ans[0])

'''
