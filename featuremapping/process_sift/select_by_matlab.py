import csv
import numpy as np

from numpy.core.fromnumeric import shape
import scipy.io as io

path1 = 'Extracted Features/names_modified.csv'
path2 = 'Extracted Features/sift_python_order/sift_file_order.csv'


python_list = []
matlab_list = []

# get current matlab list 

with open(path1) as f:
    for row in f:
        temp = row.split(',')

        matlab_list.append(str((temp[0])[:-1]))

print(matlab_list[:10])


f.close()

# get current python list
with open(path2) as f:
    for row in f:
        temp = row.split(',')
        for ele in temp:
            modify = ele[2:-4]
            python_list.append(modify)
python_list[-2] = (python_list[-2])[:-1]
python_list = python_list[:-1]
print(python_list[:10])

f.close()

# compute the difference between two lists and store the result

diff = []
for ele in python_list:
    if ele not in matlab_list:
        diff.append(python_list.index(ele))

print(diff)

diff = np.flip(diff)
print(diff.shape)


feature_idx_path = 'Extracted Features/sift_python_order/sift_histo_global.csv'

feature_idx = []
count = 0
count_name = 0
count_histo = 0
histograms = None
order= None
with open(feature_idx_path) as f:
    for row in f:
        if (count % 4) == 2:
            # print('-------------------------------------------------------------------')
            # print(len(row.split(',')[:-1].append(row.split(',')[-1][:-2])))
            temp = np.float64([row.split(',')])
            if count_histo == 0:
                histograms = temp
            else:
                histograms = np.concatenate((histograms,temp),axis=0)
            count_histo += 1
            print(histograms.shape)
           
            
            
       
     
        # feature_idx.append (row.split(',')[:-1])
        count += 1


for idx in diff:
    print(idx)
    histograms = np.delete(histograms,idx,axis=0)
    python_list = np.delete(python_list, idx)

print(histograms.shape, python_list.shape)

print()


store_sift_matlab_path = 'Extracted Features/sift_matlab_order/same_number_not_order_sifthiso.npy'
store_order_matlab_path = 'Extracted Features/sift_matlab_order/same_number_not_order_order.npy'

with open(store_sift_matlab_path, 'wb') as f:
    np.save(f,histograms)

f.close()

with open(store_order_matlab_path, 'wb') as f:
    np.save(f,python_list)

f.close()






    









