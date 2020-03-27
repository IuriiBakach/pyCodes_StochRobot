from classes_needed import *
from utilities import *
import copy
import numpy as np
import csv

# set initial parameters
robotSpeed = 3

# Create an initial matrix for shape and scale values for different zones and times
# col = zone, row = hours
los_matrix = np.zeros((1, 2), dtype='f,f').tolist()
los_matrix[0] = [(1, 1), (2, 1)]

depot = Depot(0, 1.2, 1.7)
depot.setNumberOfRobots(3)

# create customers

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

# next need to create a matrix of distances
raw_dist_matrix = all_distances([depot.getxCoord(), depot.getyCoord()], custList, [.3, .6, 1.7, 1.3])

# once I have this matrix I need to adjust all the values based on the los_matrix and reduce the matrix to the vector
# in order to have just the old distance matrix
