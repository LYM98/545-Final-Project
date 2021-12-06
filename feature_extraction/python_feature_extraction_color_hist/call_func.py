# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:28:46 2021

@author: LBY
"""

# In[1]
import try_gen_color_hist
import cv2
import numpy as np
# name = 'demo.jpg'
# rgb_hist, rgb_hist_hor, hsv_hist, hsv_hist_hor, LAB_hist, LAB_hist_hor = try_gen_color_hist.get_2_lay_out_color_features(name)

with open('names.txt', 'r') as f:
    data = f.read()
    #print(data)
    
my_names = data.split('\n')
data_to_use = my_names[0:len(my_names)-1]

base_dir = 'E:\EECS545\processed_dataset_v4\image'

length_stored_vec = 468
vec_mat = np.zeros((len(data_to_use), length_stored_vec))

idx = 0
for i in data_to_use:
    temp_name = base_dir +'\\'+ i
    #print(temp_name)
    temp_img = cv2.imread(temp_name)
    temp_img = cv2.resize(temp_img, (500, 500))
    # cv2.imshow(f'{i}', temp_img)
    
    
    ############################################
    ##### the size of the vector to store ######
    ############################################
    # rgb_hist,     48x1, vec_total[0: 48]
    # rgb_hist_hor, 36x3, vec_total[48: 156]
    # hsv_hist,     48x1, vec_total[156: 204]
    # hsv_hist_hor, 36x3, vec_total[204: 312]
    # LAB_hist,     48x1, vec_total[312: 360]
    # LAB_hist_hor  36x3, vec_total[360: 468]
    
    
    vec_total = np.zeros((1, length_stored_vec))
    rgb_hist, rgb_hist_hor, hsv_hist, hsv_hist_hor, LAB_hist, LAB_hist_hor = try_gen_color_hist.get_2_lay_out_color_features(temp_name)
    vec_total[0, 0: 48] = rgb_hist.reshape(1, 48)
    vec_total[0, 48: 156] = rgb_hist_hor.T.reshape(1, 108)
    vec_total[0, 156: 204] = hsv_hist.reshape(1, 48)
    vec_total[0, 204: 312] = hsv_hist_hor.T.reshape(1, 108)
    vec_total[0, 312: 360] = LAB_hist.reshape(1, 48)
    vec_total[0, 360: 468] = LAB_hist_hor.T.reshape(1, 108)
    
    vec_mat[idx, :] = vec_total
    idx += 1
    print(idx)
    
np.save("color_hist.npy", vec_mat)


# In[2]
import numpy as np
my_color_hist = np.load("color_hist.npy")
    
    
    
    
    




    
