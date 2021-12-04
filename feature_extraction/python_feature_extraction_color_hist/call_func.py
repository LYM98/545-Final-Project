# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:28:46 2021

@author: LBY
"""

import try_gen_color_hist
name = 'demo.jpg'
rgb_hist, rgb_hist_hor, hsv_hist, hsv_hist_hor, LAB_hist, LAB_hist_hor = try_gen_color_hist.get_2_lay_out_color_features(name)