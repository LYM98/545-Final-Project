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


if __name__ == '__main__':
    print("---- do the demo ---")
    img = cv2.imread('demo.jpg')
    set_num_bins = 12
    rgb_hist_demo = get_rgb_color_hist(img, set_num_bins)
    hsv_hist_demo = get_hsv_color_hist(img, set_num_bins)
    LAB_hist_demo = get_lab_color_hist(img, set_num_bins)
    combined_vector = combine_all(rgb_hist_demo, hsv_hist_demo, LAB_hist_demo)
    print("length of the combined vector: ", len(combined_vector))
    plt.figure()
    plt.title("combined vector hist")
    plt.xlabel("index")
    plt.ylabel("normalized features")
    plt.plot(combined_vector)
    plt.show()
