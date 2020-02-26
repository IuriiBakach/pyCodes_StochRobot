from classes_needed import *
from utilities import *
import copy
import numpy as np
import csv

# set initial parameters
robotSpeed = 3
maxTravelDist = 6

# _____parameters of the gamma distribution
shapePar = 1
scalePar = 1
# create initial data: customers, depots, routes.

# create depot with specified number of robots. Perhaps I don't really need a class-> dict would work
# a stub for future setofDepots = []

depot = Depot(0, 0, 0)
depot.setNumberOfRobots(2)

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

# create a matrix of distances from depot(s) to customers

distances = np.zeros((1, len(custList) + 1))

# ________ this returns a vector of distances form a depot to every customer in hours with robots speed;
# not round trip. robot speed to be removed when corresponding alphas are taken into account
for i in range(0, len(custList)):
    distances[0][i + 1] = (abs(custList[i].xCoord - depot.xCoord) + abs(custList[i].yCoord - depot.yCoord)) / robotSpeed

routePlan[0].insert_customer(1, custList[2], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[0], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[9], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[5], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[16], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[10], distances, shapePar, scalePar)
routePlan[0].insert_customer(1, custList[13], distances, shapePar, scalePar)

routePlan[1].insert_customer(1, custList[19], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[4], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[1], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[6], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[15], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[11], distances, shapePar, scalePar)
routePlan[1].insert_customer(1, custList[8], distances, shapePar, scalePar)

routePlan[2].insert_customer(1, custList[14], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[3], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[12], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[7], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[18], distances, shapePar, scalePar)
routePlan[2].insert_customer(1, custList[17], distances, shapePar, scalePar)

print(routePlan)

customers, objective, route = exchange(custList[1], custList[2], routePlan, distances, shapePar, scalePar)

print(route)

# final_ans = tabu_search(custList_tabu, distances, routePlan, shapePar, scalePar)
