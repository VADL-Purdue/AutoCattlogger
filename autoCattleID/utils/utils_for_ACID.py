'''
Author: Manu Ramesh

This module utility functions to help cattle identification.

'''
import cv2, pickle, pdb, glob, os, numpy as np, sys, pdb

sys.path.insert(0, '../../')
from collections import Counter
from tqdm import tqdm, trange

import logging
# from shapely.geometry import Polygon
# from multiprocessing import Pool, cpu_count
import multiprocessing
# import pandas as pd
# from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors

# import matplotlib.pyplot as plt

from autoCattlogger.helpers.helper_for_morph import morph_cow_to_template

#############################################################################################################################
############################# UTILITY FUNCTIONS #############################################################################


def getTemplateMask():
    #Fn copied over from ../bandScanner/bandScanner.py
    #Create a mask of the template cow image

    keypoint_names = kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]

    #save the template mask
    #Create a template mask by passing a white image to morph_cow_to_template function.
    #cowRotatedCropKPAligned_img, templateKPs = morph_cow_to_template(cowCropImg=cowRotatedCrop_img, warpedKPs=warpedKPs, keypoint_names = keypoint_names, selectedTriIds=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # selecting only triagles inside the cow skeleton
    templateMaskImg, _ = morph_cow_to_template(cowCropImg=np.ones((512, 256, 3), np.uint8)*255, warpedKPs=np.ones((10, 3), np.float32), keypoint_names=keypoint_names, selectedTriIds=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # selecting only triagles inside the cow skeleton

    # plt.imshow(cv2.cvtColor(templateMaskImg, cv2.COLOR_BGR2RGB))
    # plt.title(f'Template Mask Image')
    # plt.show()
    # pdb.set_trace(header='Template Mask Image')

    return cv2.cvtColor(templateMaskImg, cv2.COLOR_BGR2GRAY)  # converting to grayscale mask image


# Class to allow mutiprocessing pool inside another multiprocessing pool
# Should you call the warping function from within another multiprocessing pool, this will avoid the "daemonic processes are not allowed to have children" error.
# I was trying to run the trackWiseEvaluateAutoCattloggerV2.py which uses multiprocessing to process multiple tracks in parallel, and within that calls the graph warper which also uses multiprocessing.
# Reference: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class Undaemonize(object):
    '''Context Manager to resolve AssertionError: daemonic processes are not allowed to have children
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    
    Tested in python 3.8.5
    '''
    def __init__(self):
        self.p = multiprocessing.process.current_process()
        if 'daemon' in self.p._config:
            self.daemon_status_set = True
        else:
            self.daemon_status_set = False
        self.daemon_status_value = self.p._config.get('daemon')
    def __enter__(self):
        if self.daemon_status_set:
            del self.p._config['daemon']
    def __exit__(self, type, value, traceback):
        if self.daemon_status_set:
            self.p._config['daemon'] = self.daemon_status_value

if __name__ == "__main__":
    templateMask = getTemplateMask()
    print(f"Template mask shape: {templateMask.shape}")