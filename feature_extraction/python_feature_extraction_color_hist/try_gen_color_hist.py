# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 02:17:46 2021

@author: LBY
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


def transfer_to_n_bins(num_of_bins, original_hist):
    total_sum = np.sum(original_hist)
    # print(total_sum)
    h_groups = np.array_split(original_hist, num_of_bins)
    h_modified_hist = np.zeros([num_of_bins, 1])
    for i in range(num_of_bins):
        temp_sum = np.sum(h_groups[i])
        h_modified_hist[i, 0] = temp_sum
    return h_modified_hist / total_sum


def get_rgb_color_hist(image, num_of_bins):
    
    image_b = image[..., 0]
    image_g = image[..., 1]
    image_r = image[..., 2]
    
    hist_b = cv2.calcHist([image_b],[0],None,[256],[0,256])
    hist_b_normalized = transfer_to_n_bins(num_of_bins, hist_b)
    
    hist_g = cv2.calcHist([image_g],[0],None,[256],[0,256])
    hist_g_normalized = transfer_to_n_bins(num_of_bins, hist_g)
    
    hist_r = cv2.calcHist([image_r],[0],None,[256],[0,256])
    hist_r_normalized = transfer_to_n_bins(num_of_bins, hist_r)
    
    rgb_color_hist = np.vstack((hist_b_normalized, hist_g_normalized, hist_r_normalized))
    return rgb_color_hist
    
    

def get_hsv_color_hist(image, num_of_bins):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h = img_hsv[..., 0]
    img_s = img_hsv[..., 1]
    img_v = img_hsv[..., 2]
    
    hist_h = cv2.calcHist([img_h],[0],None,[181],[0,181])
    hist_h_normalized = transfer_to_n_bins(num_of_bins, hist_h)
    
    hist_s = cv2.calcHist([img_s],[0],None,[256],[0,256])
    hist_s_normalized = transfer_to_n_bins(num_of_bins, hist_s)
    
    hist_v = cv2.calcHist([img_v],[0],None,[256],[0,256])
    hist_v_normalized = transfer_to_n_bins(num_of_bins, hist_v)
    
    hsv_color_hist = np.vstack((hist_h_normalized, hist_s_normalized, hist_v_normalized))
    return hsv_color_hist
    
 
def get_lab_color_hist(image, num_of_bins):
    img_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_L = img_LAB[..., 0]
    img_A = img_LAB[..., 1]
    img_B = img_LAB[..., 2]
    
    hist_L = cv2.calcHist([img_L],[0],None,[181],[0,181])
    hist_L_normalized = transfer_to_n_bins(num_of_bins, hist_L)
    
    hist_A = cv2.calcHist([img_A],[0],None,[256],[0,256])
    hist_A_normalized = transfer_to_n_bins(num_of_bins, hist_A)
    
    hist_B = cv2.calcHist([img_B],[0],None,[256],[0,256])
    hist_B_normalized = transfer_to_n_bins(num_of_bins, hist_B)
    
    LAB_color_hist = np.vstack((hist_L_normalized, hist_A_normalized, hist_B_normalized))
    return LAB_color_hist
    


def combine_all(rgb_hist, hsv_hist, LAB_hist):
    combined_vector = np.vstack((rgb_hist, hsv_hist, LAB_hist))
    return combined_vector


def get_2_lay_out_color_features(img_name):
    # print("---- do the demo ---")
    img = cv2.imread(img_name)
    img = cv2.resize(img, (500, 500))
    img_g = np.array_split(img, 3)
    img_hor_1 = img_g[0]
    img_hor_2 = img_g[1]
    img_hor_3 = img_g[2]
    # cv2.imshow(img_name + 'out1', img_hor_1)
    # cv2.imshow(img_name + 'out2', img_hor_2)
    # cv2.imshow(img_name + 'out3', img_hor_3)
    
    set_num_bins_whole = 16
    set_num_bins_hor = 12
    
    # get the global layout
    rgb_hist = get_rgb_color_hist(img, set_num_bins_whole)
    hsv_hist = get_hsv_color_hist(img, set_num_bins_whole)
    LAB_hist = get_lab_color_hist(img, set_num_bins_whole)
    
    # get the 3x1 layout of rgb
    rgb_hist_hor = np.zeros([set_num_bins_hor * 3, 3])
    rgb_hist_hor_1 = get_rgb_color_hist(img_hor_1, set_num_bins_hor)
    rgb_hist_hor_2 = get_rgb_color_hist(img_hor_2, set_num_bins_hor)
    rgb_hist_hor_3 = get_rgb_color_hist(img_hor_3, set_num_bins_hor)
    rgb_hist_hor[:, 0] = rgb_hist_hor_1[:, 0]
    rgb_hist_hor[:, 1] = rgb_hist_hor_2[:, 0]
    rgb_hist_hor[:, 2] = rgb_hist_hor_3[:, 0]
    
    # get the 3x1 layout of hsv
    hsv_hist_hor = np.zeros([set_num_bins_hor * 3, 3])
    hsv_hist_hor_1 = get_hsv_color_hist(img_hor_1, set_num_bins_hor)
    hsv_hist_hor_2 = get_hsv_color_hist(img_hor_2, set_num_bins_hor)
    hsv_hist_hor_3 = get_hsv_color_hist(img_hor_3, set_num_bins_hor)
    hsv_hist_hor[:, 0] = hsv_hist_hor_1[:, 0]
    hsv_hist_hor[:, 1] = hsv_hist_hor_2[:, 0]
    hsv_hist_hor[:, 2] = hsv_hist_hor_3[:, 0]
    
    # get the 3x1 layout of LAB
    LAB_hist_hor = np.zeros([set_num_bins_hor * 3, 3])
    LAB_hist_hor_1 = get_lab_color_hist(img_hor_1, set_num_bins_hor)
    LAB_hist_hor_2 = get_lab_color_hist(img_hor_2, set_num_bins_hor)
    LAB_hist_hor_3 = get_lab_color_hist(img_hor_3, set_num_bins_hor)
    LAB_hist_hor[:, 0] = LAB_hist_hor_1[:, 0]
    LAB_hist_hor[:, 1] = LAB_hist_hor_2[:, 0]
    LAB_hist_hor[:, 2] = LAB_hist_hor_3[:, 0]
    
    
    # show the combined vector of global layout
#    combined_vector = combine_all(rgb_hist, hsv_hist, LAB_hist)
#    print("length of the combined vector: ", len(combined_vector))
#    plt.figure()
#    plt.title("combined vector hist")
#    plt.xlabel("index")
#    plt.ylabel("normalized features")
#    plt.plot(combined_vector)
#    plt.show()
    return rgb_hist, rgb_hist_hor, hsv_hist, hsv_hist_hor, LAB_hist, LAB_hist_hor
    

if __name__ == "__main__":
    file_name = 'demo.jpg'
    rgb_hist, rgb_hist_hor, hsv_hist, hsv_hist_hor, LAB_hist, LAB_hist_hor = get_2_lay_out_color_features(file_name)

    
