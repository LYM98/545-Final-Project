import csv
import numpy as np
from numpy.lib.twodim_base import diagflat
import scipy.io as io
feature_idx_path = 'Extracted Features/sift_histo_global.csv'

feature_idx = []
count = 0
count_name = 0
count_histo = 0
histograms = None
order= None
with open(feature_idx_path) as f:
    for row in f:
        if (count % 4) == 2:
            print('-------------------------------------------------------------------')
            # print(len(row.split(',')[:-1].append(row.split(',')[-1][:-2])))
            temp = np.float64([row.split(',')])
            if count_histo == 0:
                histograms = temp
            else:
                histograms = np.concatenate((histograms,temp),axis=0)
            count_histo += 1
            print(histograms.shape)
           
            

        if (count % 4) == 0:
            temp = np.array([row.split(',')])
            if count_name == 0:
                order = temp
            else:
                order = np.concatenate((order,temp))
            count_name += 1
            print(order.shape)
            
       
     
        # feature_idx.append (row.split(',')[:-1])
        count += 1

# io.savemat('sift_histo.mat', {'HISTO':histograms,'order': order})

f.close()
d = open('order.csv', 'w', encoding='UTF8')   
writer = csv.writer(d)
writer.writerow(order)
d.close()


# feature_idx = feature_idx[0]
