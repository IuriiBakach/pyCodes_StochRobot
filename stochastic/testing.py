from classes_needed import *
from utilities import *
import copy
import numpy as np
import csv

"""
read in customers line by line 
if a customer is inside the zone the time windows should be changed in the following manner:

upper is increased by 1 if it is not 8
otherwise decrease lower by 1

write it into the same csv file
"""

# [xLow, yLow, xUp, yUp]
zoneCoords = [.3, .6, 1.7, 1.3]

custList = []
with open('customersTest.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        custList.append(Customer(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

for elem in custList:
    if zoneCoords[0] < elem.xCoord < zoneCoords[2] and zoneCoords[1] < elem.yCoord < zoneCoords[3]:
        if elem.lateTW < 8:
            elem.lateTW += 1
        else:
            elem.earlyTW -= 1

# now all the tws are modified. The next step is to override the csv file

with open('customersTestWriter.csv', mode='w', newline='') as csv_file:
    fieldnames = ['id', 'xCoord', 'yCoord', 'eTW', 'lTW']
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    for elem in custList:
        writer.writerow([elem.id, elem.xCoord, elem.yCoord, elem.earlyTW, elem.lateTW])

csv_file.close()
