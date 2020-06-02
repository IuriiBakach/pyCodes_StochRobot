import csv
import numpy as np

from classes_needed import *

# def main():
# set initial parameters
robotSpeed = 3

# Create an initial matrix for shape and scale values for different zones and times
# col = zone, row = hours
los_matrix = np.zeros((1, 2), dtype='f,f').tolist()

# _____parameters of the gamma distribution; order -> shape outer zone, shape inner zone
los_matrix[0] = [(1, 1), (4, 1)]
# these are just stub values, potentially to update later

print("Outer zone shape is {} and inner zone shape is {}".format(los_matrix[0][0][0], los_matrix[0][1][0]))

# this approach works only with the case if I am not to touch scale which is bad and I need to rework it in the way
# scale parameter can be varied and taken into account

shapePar = 1
scalePar = 1
# create initial data: customers, depots, routes.

# how about create a list with a set of csv files and then run the alg over all of them
csv_list = ['1.csv', '2.csv']  # , '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv']

for item in csv_list:
    # read in customers
    custList = []
    with open(item, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            custList.append(Customer(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    # create depot with specified number of robots. Perhaps I don't really need a class-> dict would work

    depot = Depot(0, 0, 0)
    depot.setNumberOfRobots(2)

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
    # now I need to fill initial routes with customers
    print(routePlan[0])
    
    routePlan[0].insert_customer(2, custList[3], distances, shapePar, scalePar)
    print(routePlan[0])
    
    routePlan[0].insert_customer(2, custList[0], distances, shapePar, scalePar)
    print(routePlan[0])  
    '''

    routePlan[0].insert_customer(1, custList[0], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[1], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[2], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[3], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[4], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[5], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[6], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[7], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[8], distances, shapePar, scalePar)
    routePlan[0].insert_customer(1, custList[9], distances, shapePar, scalePar)

    routePlan[1].insert_customer(1, custList[10], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[11], distances, shapePar, scalePar)
    routePlan[1].insert_customer(1, custList[12], distances, shapePar, scalePar)
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

# if __name__ == "__main__":
#    main()

"""
how do I model the change of initial departure time? essentially it just the arrival time to the first customer.

one way is to add a dummy customer to all routes into the first position that will guarantee that the arrival time to
the actual customer 1 happens later. I think I'll do this

how to proceed witt shape/scale revision. Whenever I initially compute best paths I need to return 
(distance, resulting shape, scale). In order to get mean travel time I need to multiply everything. However I need 
only distance * shape for computations and scale separately. That being said, whenever I add/remove customer
I can use distance*shape and that would be the content of self.distances

or those can be only distances I compute. So I return to the triplet. Do I also need a cumulative shape then?
I guess I do

Ok, so how do I do it? 
step 1: compute resulting paths as (distance, resulting shape, scale)
step 2: redo insert/remove customer from a route so the 3 independent numbers are being tracked : total distance, 
total shape and scale(?) instead of just distance
step 3: update computations for earliness/lateness 

there is a questions of how to get a combined shape if for all of the chuncks it's different? how about this approach: 
whenever I add a customer I just add it to the list with corresponding id without any dist/whatever recomputations. 
At this point whenever I add a cust I have only a route with custs, and their corresp data. Then, all earliness and
lateness in getting computed in the separate function and just returned separately using distances and shapes. 

Similar procedure with removal: find the position, remove cust with proper dist and shape and compute ealiness and
lateness. Potentially I can get some computations into the earliness/lateness function outside to decrease a number of 
operations 

How would shifting work from now? If I do it this way I'll just have to add to an array of shape coeffs a fixed number
that also depends on the scale parameter so that mean arrival time is shifted correspondingly
(which is, I assume, n/scale)

things still left to do:

redo the simplified version of customer_insert and customer_delete
DEBUG!!!11
how to debug: 
1) check if paths are computed correctly
2) check if cust addition is rights
3) check if obj function is right


add waiting. Think about it can be done
"""
