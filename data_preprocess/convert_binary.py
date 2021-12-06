import numpy as np
# import nltk
# import xml.etree.ElementTree as ET
# import os
# from bs4 import BeautifulSoup
# import enchant
import csv
# import inflect



# feature_path = 'matlab_order_features/feature.csv'
# feature_idx_path = 'matlab_order_features/feature_index.csv'

# feature_idx = []
# with open(feature_idx_path) as f:
#     for row in f:
        
#         feature_idx.append (row.split(','))



# # feature_idx[0]='light'
# feature_idx[0][0]= 'light'
# feature_idx[0][-1] = 'contemplation'
# print(feature_idx[0])
# feature_idx = feature_idx[0]
# print(len(feature_idx))
# f.close()

# counter = 0
# binary_feature = None
# string_features = np.load('matlab_order_features/feature_ordered.npy',allow_pickle=True)
# for ele in string_features:
    



#     binary_row = np.zeros((1,len(feature_idx)))
#     # if counter % 2 == 0:
#     #     temp = row.split(',')
#     #     try: 
#     #         idx = temp.index('')
#     #         temp = temp[:idx]
#     #     except:
#     #         pass
#     for i in ele:
#         idx = feature_idx.index(i)
#         binary_row[0,idx] = 1
    
         
#     if counter==0:
#         binary_feature = binary_row

#     else:

#         binary_feature = np.concatenate((binary_feature,binary_row),axis=0)

 
#     counter +=1
#     print(binary_feature.shape)
        




# np.save('matlab_order_features/binary_feature.npy', binary_feature)

data=np.load('matlab_order_features/binary_feature.npy')
print(data.shape)



    












