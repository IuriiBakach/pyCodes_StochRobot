from classes_needed import *
from utilities import *
import copy
import numpy as np
import csv
'''
customer1 = Customer(1, 0.53, 0.67, 485, 545)
customer2 = Customer(2, 1.17, 1.23, 530, 590)
customer3 = Customer(3, 0.83, 1.67, 720, 780)
customer4 = Customer(4, 1.35, 1.83, 640, 700)
customer5 = Customer(5, 0.13, 0.57, 900, 960)
customer6 = Customer(6, 1.36, 0.45, 800, 860)

custList = [customer1, customer2, customer3, customer4, customer5, customer6]

routePlan = []

depot = Depot(0, 0, 0)

distances = np.zeros((1, len(custList) + 1))

# ________ this returns a vector of distances form a depot to every customer in minutes; not round trip
for i in range(0, len(custList)):
    distances[0][i + 1] = (abs(custList[i].xCoord - depot.xCoord) + abs(custList[i].yCoord - depot.yCoord)) * 60

# create a list of routes with the number of empty routes corresponding to the number of robots in the depot
for elem in range(2):
    routePlan.append(Route(elem))

routePlan[0].insert_customer(1, customer1, distances, 1, 1)
routePlan[0].insert_customer(2, customer3, distances, 1, 1)
routePlan[0].insert_customer(1, customer5, distances, 1, 1)

routePlan[1].insert_customer(1, customer2, distances, 1, 1)
routePlan[1].insert_customer(2, customer4, distances, 1, 1)
routePlan[1].insert_customer(1, customer6, distances, 1, 1)

totalObj = 0

for route in routePlan:
    totalObj += route.total_earliness()
    totalObj += route.total_lateness()
#routePlan_toIns = copy.deepcopy(routePlan)

# when I do alg, need to put flags for each different reloc operator to keep track

move, modifiedObj, modifiedRP = exchange(customer1, customer2, routePlan, distances, 1, 1)

#routePlanMod_1, modifiedObj_1= one_shift(customer1, customer2, routePlan_toIns, distances, 1, 1)

print("here")
'''

customerList = []
with open('customers.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        customerList.append(Customer(int(row[0]), float(row[1]), float(row[2]), int(row[3]), int(row[4])))


