import random
import csv

random.seed(1)

# what do I need to do?
# keep creating customers 10 times
# inside every iteration till x cust is added I need to randomly create coord and check where they fall in

custsets = []
max_len = 20

# zone [.3, .6, 1.7, 1.3]
# depot [1.2,1.7]

comb_cust_set = [0] * 10
tmp_set = []

for i in range(0, 10):
    tmp_cust_set_in = []
    tmp_cust_set_out = []

    # print([round(random.random(), 1), round(random.random(), 1)])

    while len(tmp_set) < 50:
        # create a customer, if in the zone, ad to one set, else, to the second after checking for duplicates.
        # ones enough are created, concatenate sets and
        tmp_set = []

        cust = [round(random.randrange(0, 20, 1) / 10, 1), round(random.randrange(0, 20, 1) / 10, 1)]
        # print(cust)

        # in zone cust
        if .3 < cust[0] < 1.7 and .6 < cust[1] < 1.3 and len(tmp_cust_set_in) < 17:
            if cust not in tmp_cust_set_in:
                tmp_cust_set_in.append(cust)

        # out zone cust
        else:
            if cust not in tmp_cust_set_out and cust is not [1.2, 1.7] and len(tmp_cust_set_out) < 33:
                tmp_cust_set_out.append(cust)

        tmp_set = tmp_cust_set_in + tmp_cust_set_out

    comb_cust_set[i] = tmp_cust_set_in + tmp_cust_set_out
    tmp_set = []

    # need to write this into a csv file

for i in range(0, 10):
    file_name = 'new_data_c_50_p_35' + '_' + str(i + 1) + '.csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(comb_cust_set[i])
        file.close()
