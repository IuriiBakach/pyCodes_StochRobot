from utilities import *


class Depot:
    def __init__(self, id, xCoord, yCoord):
        self.id = id
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.numberOfRobots = 0

    def getId(self):
        return self.id

    def getxCoord(self):
        return self.xCoord

    def getyCoord(self):
        return self.yCoord

    def getNumberRobots(self):
        return self.numberOfRobots

    def setNumberOfRobots(self, number):
        self.numberOfRobots = number


class Customer:

    def __init__(self, id, xCoord, yCoord, earlyTW, lateTW):
        self.id = id
        self.xCoord = xCoord
        self.yCoord = yCoord

        # _________ Time window is in minutes from 0 to 60*24

        self.earlyTW = earlyTW
        self.lateTW = lateTW

    def getId(self):
        return self.id

    def getXCoord(self):
        return self.XCoord

    def getYCoord(self):
        return self.YCoord

    def getEarlyTW(self):
        return self.earlyTW

    def getLateTW(self):
        return self.lateTW

    def __str__(self):
        return str(self.getId())

    def __repr__(self):
        return str(self.getId())


class Route:
    def __init__(self, id):
        self.id = id
        self.currentRoute = []
        # _________add a depot as a starting location_________
        depot = Depot(0, 0, 0)
        self.currentRoute.append(depot)
        # _________ total distance a robot travelled on this route_____
        # _________ add 0-s since the first location in the depot for every route
        self.distances = []
        self.distances.append(0)
        self.earliness = []
        self.earliness.append(0)
        self.lateness = []
        self.lateness.append(0)

    def find_customer_pos(self, customer):

        # perhaps I'll have to rework this stuff

        for index, elem in enumerate(self.currentRoute):
            if elem.getId() == customer.getId():
                return index
        '''
        try:
            return self.currentRoute.index(customer)
        except ValueError:
            pass
        '''

    def insert_customer(self, pos, cust, distMatr, shape, scale):
        # __________ insert a customer into a specific position in the route
        # __________ distances is a matrix of distances from a depot to all customers
        # index starts from 0

        self.currentRoute.insert(pos, cust)

        # __________ earliness/delays for every cust should be recomputed
        # ___________early and late TW for every cust in the routes are to be recomputed.I'll leave that for now

        # add distance to the customer into the list of distances
        self.distances.insert(pos, distMatr[0][cust.getId()])

        # update all the remaining distances: update inserted by adding previous and after that update all the
        # subsequent by adding inserted
        # all distances reported are distances to get to customer after visiting all the previous customers,
        # taking into account roundtrips

        # ______ if initially the route is not empty

        # tmp =  self.currentRoute[pos - 1].getId()

        if self.distances[pos - 1] != 0:
            change = self.distances[pos] + self.distances[pos - 1] + distMatr[0][self.currentRoute[pos - 1].getId()]
            self.distances[pos] = change
        else:
            self.distances[pos] += self.distances[pos - 1]

        # update all the subsequent numbers
        if pos + 1 != len(self.distances):
            for i, elem in enumerate(self.distances[pos + 1:], 1):
                self.distances[pos + i] = elem + 2 * distMatr[0][cust.getId()]
        # ______ if initially the route is empty
        elif self.distances[pos - 1] == 0:
            self.distances[pos] = distMatr[0][cust.getId()]

        # ______ inserting in the end of the sequence
        else:
            self.distances[pos] = self.distances[pos - 1] + distMatr[0][cust.getId()] + distMatr[0][
                self.currentRoute[-2].id]

        self.earliness = earliness_array(self.distances, self.currentRoute, shape, scale)
        self.lateness = lateness_array(self.distances, self.currentRoute, shape, scale)

    def remove_customer(self, pos, distMatr, shape, scale):
        # _________ remove a customer from the position specified
        # also need to update corresponding distances, earliness and lateness

        # before I remove that customer I need to know the distance to it
        distance_to_removed_customer = distMatr[0][self.currentRoute[pos].getId()]

        # now I can safely remove it
        self.currentRoute.pop(pos)  # pop starts form a 0-position

        # I recompute distances by substructing from all subsequent customers the time it take to get and come back
        # to the removed customer, unless it's on the last position
        self.distances.pop(pos)
        if pos != len(self.currentRoute) + 1:
            for i, elem in enumerate(self.distances[pos:], 0):
                # need to substruct travel time
                self.distances[pos + i] = self.distances[pos + i] - 2 * distance_to_removed_customer

        # ________earliness
        # for every element form the matrix of distance, use that elem to compute corresponding earliness
        self.earliness.pop(pos)
        self.lateness.pop(pos)

        # recompute earliness and lateness
        if len(self.currentRoute) > 1:
            self.earliness = earliness_array(self.distances, self.currentRoute, shape, scale)
            self.lateness = lateness_array(self.distances, self.currentRoute, shape, scale)

    def total_earliness(self):
        # _________ compute a total earliness for a route
        total = 0
        for elem in self.earliness:
            total += elem
        return total

    def total_lateness(self):
        # _________compute a total lateness for a route
        total = 0
        for elem in self.lateness:
            total += elem
        return total

    def __str__(self):
        # __________ need to modify this to make it more readable
        # return str(self.id) + str(self.currentRoute) + str(self.distances) + str(self.earliness) + str(self.lateness)
        return_string = ' '
        for c in self.currentRoute:
            return_string += str(c.getId())
            return_string += ' ->'
        return return_string

    def __repr__(self):
        # I want to return a route : 0-> 1-> etc.
        return_string = ' '
        for c in self.currentRoute:
            return_string += str(c.getId())
            return_string += ' ->'
        return return_string
