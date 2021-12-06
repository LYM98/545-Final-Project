import numpy as np
import scipy.io as io
path1 = 'Extracted Features/names_modified.csv'

store_sift_matlab_path = 'Extracted Features/sift_matlab_order/same_number_not_order_sifthiso.npy'
store_order_matlab_path = 'Extracted Features/sift_matlab_order/same_number_not_order_order.npy'

matlab_list = []
with open(path1) as f:
    for row in f:
        temp = row.split(',')

        matlab_list.append(str((temp[0])[:-1]))




f.close()


histograms = np.load(store_sift_matlab_path)

python_list = np.load(store_order_matlab_path)




rearr_idx = []
for ele in matlab_list:
    idx = np.where(python_list==ele)
    
    rearr_idx.append((idx[0])[0])

histograms = histograms[rearr_idx,:]
python_list = python_list[rearr_idx]

io.savemat('Extracted Features/sift_matlab_order/sift_histo.mat', {'HISTO':histograms,'order': python_list})

store_sift_matlab_path1 = 'Extracted Features/sift_matlab_order/same_number_order_sifthiso.npy'
store_order_matlab_path1 = 'Extracted Features/sift_matlab_order/same_number_order_order.npy'

with open(store_sift_matlab_path1, 'wb') as f:
    np.save(f,histograms)

f.close()

with open(store_order_matlab_path1, 'wb') as f:
    np.save(f,python_list)

f.close()