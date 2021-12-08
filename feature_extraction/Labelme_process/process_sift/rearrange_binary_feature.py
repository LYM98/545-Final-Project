import numpy as np
import scipy.io as io
import csv

# store_order_matlab_path = 'Extracted Features/sift_matlab_order/same_number_order_order.npy'  

# order= np.load(store_order_matlab_path)
# count = 0
# binary=[]
# path = 'processed_dataset_v4/binary_feature.csv'
# with open(path) as f:
#     for row in f:
        
#         temp = row.split(',')
#         # print((temp[0])[:-4]+'.jpg')
#         if (count%2) == 0:
#             binary.append(temp)
#         count+=1


# binary_ordered = []
# count = 0
# store_order_matlab_path1 = 'Extracted Features/sift_matlab_order/binary_feature_ordered_matlab.csv'
# f = open(store_order_matlab_path1, 'w', encoding='UTF8')
# writer = csv.writer(f)
# for ele in order:
#     for b in binary:

    
#         k = (b[0])[:-4]+'.jpg'
     
#         if k == ele:
#             writer.writerow(b)
#             count+=1
#             print(count)

# f.close()
            


# store_order_matlab_path1 = 'Extracted Features/sift_matlab_order/binary_feature_ordered_matlab.csv'
# features = []
# count = 0
# with open(store_order_matlab_path1) as f:
#     for row in f:     
#         if (count%3) == 0:
#             temp = row.split(',')
            
#             temp[-1] = (temp[-1])[1:]
    
#             temp1 = temp[1:]
#             results = list(map(int, temp1))       
#             features.append(results)
#             print(len(features))
#         count+=1


      
  
    


        
       
       
        