'''
Author: Manu Ramesh

This is to automatically register cows into the cattlog by looking at cut-videos.
Additional funtionality may be added to register cows from un-cut raw videos.
'''


import torch
import torchvision.models as models
import torch.nn as nn
import pickle, yaml
from PIL import Image

from collections import Counter

import sys
sys.path.insert(0, '../') #top level directory

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog

frame_width = 2304 #int(cap.get(3))
frame_height = 1296 #int(cap.get(4))
#out = cv2.VideoWriter('infout.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

import pdb, glob
from multiprocessing import Pool

from autoCattlogger.utils.display_utils import get_cow_autocattlog_frame

#COW RECOGNITION - depricated
#from trial2_bodies import make_recognition_model
#from trial2_bodies import get_test_transform

from torchvision import transforms
from PIL import Image

from autoCattlogger.helpers.helper_for_infer import draw_and_connect_keypoints
from autoCattlogger.helpers.helper_for_infer import cow_tv_KP_rule_checker
from autoCattlogger.helpers.helper_for_infer import cow_tv_KP_rule_checker2
from autoCattlogger.helpers.helper_for_infer import cow_tv_KP_rule_checker_auxiliary
from autoCattlogger.helpers.helper_for_infer import get_visible_KPs
from autoCattlogger.helpers.helper_for_infer import log_cow_det_reid_stat
from autoCattlogger.helpers.helper_for_morph import morph_cow_to_template
from autoCattlogger.helpers.helper_for_kpInterpolation import interpolateKPs
from autoCattlogger.helpers.helper_for_QR_and_inference import inferImage
from autoCattlogger.helpers.helper_for_tracking import getNewTrackPoint, getNewTrack, matchTrackPoints

#from experiment_hierarchical_mapper import inferImage #old file

from autoCattlogger.helpers.helper_for_kpCorrection import correctMispalcedKPs_iterativeMode, correctMispalcedKPs_StrategicBruteForceMode, computeWeightsForMisplacedKPs

import logging
from tqdm import tqdm, trange
import warnings
warnings.simplefilter('ignore', np.RankWarning) #ignoreing warnings here. Comment this line if you want to see the "RankWarning: Polyfit may be poorly conditioned" warning

from autoCattlogger.helpers.helper_for_colorCorrection import correctPatchWiseBlackPointsOnFramePS1 #for black point color correction

#from copy import deepcopy #for copying dictionary -> self.__dict__ for logging purposes

#mmpose for keypoints
from mmpose.apis import init_model, inference_topdown
#from ..Experiments.mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

#for parallel evaluation of videos
from joblib import Parallel,delayed

from autoCattlogger.helpers.helper_for_QR_and_inference import pixelate

from autoCattlogger._autoCattloggerBase import _AutoCattloggerBase
from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors

from shapely.geometry import Polygon #for finding IOUs of cows
import pandas as pd

#for matching trackPoints to tracks of multiple cows
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching, maximum_bipartite_matching


#for postprocessing tracks
from autoCattlogger.helpers.helper_for_autoCattlog import postProcessTracks1, postProcessTracks_RL

#for reading frames from hdf5 files
from autoCattlogger.helpers.helper_for_hdf5 import VideoCapture_HDF5


class AutoCattloger(_AutoCattloggerBase):

    def __init__(self,*args, **kwargs):
        # super().__init__(**kwargs)
        super().__init__(*args, **kwargs)
    
        #just so that other functions can use these values
        self.predictions = None
        self.rotatedBboxPtsList = []



    def getBitVectorsFromFrame(self, frame, countsDict, kpRulesPassCounts, nKPDetectionCounts, frameCount=0, gt_label=None, doNotComputeCowAutoCatOutImg=False):
        '''

        Gets the bit vector (feature vector) of the cow in the frame.

        *******************
        This method is now modified to work with multiple cows in the frame. 
        It returns the bit vectors and template aligned images of all the cows in the frame.
        *******************

        :param frame:
        :param countsDict:
        :param kpRulesPassCounts:
        :param nKPDetectionCounts:
        :param frameCount:
        :param gt_label:
        :return: bitVecStrList
        '''

        bitVecStrList = [] #to store the list of computed bitVectors. Should essentially contain only one entry as we are processing only one cow per frame.
        bitVecStr = '' #the bit vector of the given cow

        kpCorrectionMethodList = [] #to store the keypoint correction (KPC) method used. If no correction was used, None value will be stored. This is to allow use to selectively use instances based on whether KPC was used or not -- for ablation study in the paper.

        #resetting the values before processing the frame
        self.predictions = None
        self.rotatedBboxPtsList = []

        #DETECTRON2
        predicted_cowID = None  # for returning, for single cow per frame mode

        frameSafe = frame.copy()  # so that we do not write anything on this frame
        
        #Instead of color-correcting every frame, I have moved this functionality below so that we color-correct only when there is a cow in the frame. This is to save some computation.
        #if self.colorCorrectionOn:
        #    if self.externalCC_fn is not None:
        #        frameCC = self.externalCC_fn(frame)
        #    else: #our method
        #        frameCC = correctPatchWiseBlackPointsOnFramePS1(frameSafe, blackPointImg=self.blackPointImg, patchSize=1)  # Color Corrected Frame
        #else:
        #    frameCC = None
        frameCC = None # we now update it only if it is not None. To avoid redundant computation and save time.

        frameH, frameW, _ = frame.shape

        outputs = self.predictor(frame)
        # print(f"outputs = \n{outputs}")

        predictions = outputs['instances'].to('cpu')
        self.predictions = predictions #so that other functions can use these values
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None

        num_instances = boxes.shape[0]
        # print(f"\n\n__dict__ = {outputs['instances'].__dict__}\n\n")
        # print(f"\n\nnumber of instances = {outputs['instances'].to('cpu').pred_boxes.tensor.numpy().shape}\n\n")

        # Don't save frames that don't have even a single predicted cow
        if num_instances < 1:
            # continue - not in a loop any more
            # print(f"Number of instances < 1. Returning!")
            # logging.debug(f"Number of instances < 1. Returning!")
            return None  # see if you can change this later

        countsDict['total_framesWithAtLeaset1bbox'] += 1
        logging.debug(f"\n\nProcessing prediction frame {frameCount}")
        # print(f"Predictions = {predictions}")

        # https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
        # and from code of Visualizer

        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        #keypoints = [] #Need not pre-declare in python if all paths in if-else ladder defines this variable.
        #MMPOSE
        if self.useMMPOSE:
            #pose_results = inference_topdown(pose_estimator, frame, bboxes=[(245, 5, 1193, 462), (878, 2, 1817, 681), (1418, 93, 1919, 707)]) # An example
            pose_results = inference_topdown(self.pose_estimator, frame, bboxes=boxes)
            data_samples = merge_data_samples(pose_results)  # Merge the given data samples into a single data sample.
            # we need to merge data so that we get all keypoints in a single numpy array. This is to make it compatible with the rest of the code that I have already written.
            keypoints = np.dstack((data_samples.pred_instances.keypoints, data_samples.pred_instances.keypoints_visible)) # dstack -> depth stack - concatenation along third dimension. Of the shape -> (nInstances X nKeypoints X 3) where 3 is from x, y, visibility_score
        #DETECTRON2
        else:
            keypoints = predictions.pred_keypoints.numpy() if predictions.has("pred_keypoints") else None


        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            # masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self.printDebugInfoToScreen: print(f"number of masks in the frame = {masks.shape[0]}")
        # pdb.set_trace()

        rotatedBBOX_img = []
        cowRotatedCropKPAligned_img = [] #just to save and return the last available image

        # for cow recognition evaluation
        cropList = []

        for maskNumber in range(masks.shape[0]):

            # remove later if  not needed
            if not masks[maskNumber, :, :].any():
                # this is the case where the model outputs a binary mask image but all values in the image are 0 (False).
                # this freak case occured when running surabhi experiments
                if self.printDebugInfoToScreen: print(f"MaskRCNN Output is all 0 mask.")
                logging.debug(f"MaskRCNN Output is all 0 mask.")

                # reduce number of instances by 1
                num_instances -= 1
                if self.printDebugInfoToScreen: print(
                    f"As mask had no pixel on, reducing the total number of instances by 1. Current total number of instances = {num_instances}.")
                logging.debug(
                    f"As mask had no pixel on, reducing the total number of instances by 1. Current total number of instances = {num_instances}.")
                # pdb.set_trace()

                if num_instances == 0:
                    return None  # copying what I am already doing above
                    # this way, we do not have to deal with sending empty image to cow get_cow_autocattlog_frame() function below.
                else:
                    continue

            countsDict['total_instanceBBoxes'] += 1
            logging.debug(f"Mask number {maskNumber}")

            correctedKPs = []  # for writing to the hard examples info file - list of NAMES of corrected keypoints - not the location coordinates

            mask_img = masks[maskNumber, :, :].astype(np.uint8)  # *255

            # dialation (may not be necessary)
            _, mask_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            mask_img = cv2.dilate(mask_img, kernel, iterations=1)

            if not self.colorCorrectionOn:
                overlap = cv2.bitwise_and(frameSafe, frameSafe, mask=mask_img)  # original - even in SURABHI
            else:
                # trying color correction

                #color correcting frame only when there is a cow in the frame. To save some computation
                #also, we do this only if the color corrected frame (frameCC) has not been computed before
                if frameCC is None:
                    if self.externalCC_fn is not None:
                        frameCC = self.externalCC_fn(frame)
                    else: #our method
                        frameCC = correctPatchWiseBlackPointsOnFramePS1(frameSafe, blackPointImg=self.blackPointImg, patchSize=1)  # Color Corrected Frame
                
                overlap = cv2.bitwise_and(frameCC, frameCC, mask=mask_img)  # for project cow lighting - colorCorrection/colorConstancy

            # assuming contours are continuous as the cows are assumed to be in full view without any occlusion
            # contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 1:
                if self.printDebugInfoToScreen: print(
                    f"More than one contour detected. Not Breaking. Using largest contour.")
                logging.debug(f"More than one contour detected. Not Breaking. Using largest contour.")
                contours = sorted(contours, key=cv2.contourArea)[::-1]  # sort in descending order
                # cv2.imwrite(f"multi_contour_image_{frameCount}.jpg", overlap)
                # break

            # remove later if not needed
            # elif len(contours) == 0: #occured in one case when running surabhi experiments
            #    # keypoint rcnn detects mask but all points are false
            #    print(f"No contour detected. Skipping this mask. Current cow gt_label: {gt_label}")
            #    logging.debug(f"No contour detected. Skipping this mask. Current cow gt_label: {gt_label}")
            #    pdb.set_trace()
            #    continue
            # print(f"DID NOT CONTINUE")

            cnt = contours[0]  # assuming only one as maskRCNN has not been trained on  merged annotations, choosing largest one as maskRCNN is inferring multiple contours even though it is not trained that way
            # print(f"contours = {contours}") #uncomment for debugging

            #DISCONTINUED
            # if ROI_mask_path != None:
            #    M = cv2.moments(cnt)
            #    #print(f"moments = {M}")
            #
            #    #centroid
            #    cx = int(M['m10'] / M['m00'])
            #    cy = int(M['m01'] / M['m00'])
            #    print(f"Centroid cx = {cx}, cy = {cy}")
            #    if ROI_maskImg[cy, cx] == 0:
            #        #outside ROI
            #        print(f"\nDetected cow outside ROI. Not saving image!\n")
            #        continue

            # draw rotated rectangle
            # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            '''
            Here, bounding rectangle is drawn with minimum area, so it considers the rotation also.The function used is cv2.minAreaRect().
            It returns a Box2D structure which contains following detals - ( top-left corner(x, y), (width, height), angle of rotation ).
            But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv2.boxPoints()
            '''
            rect = cv2.minAreaRect(cnt)
            boxAngle = rect[-1]  # angle that rectangle makes with horizontal, in degrees

            if self.printDebugInfoToScreen: print(f"Rect rotated = {rect}")
            box = cv2.boxPoints(rect)  # 4 x 2 np array [[xA, yA], [xB, yB], [xC, yC], [xD, yD]
            if self.printDebugInfoToScreen: print(f"box before = {box}")
            '''
            A--B
            |  |
            D--C

            Angle is angle of AD with horizontal, with angle at D
            '''
            box = np.intp(box) #np.int0(box)  # converting to int64
            self.rotatedBboxPtsList.append(box)

            rotatedBBOX_img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)  # comment, otherwise next mask's images will have drawing

            # labeling the points
            #font = cv2.FONT_HERSHEY_SIMPLEX;
            #fontScale = 1;
            #fontColor = (255, 255, 255);
            #thickness = 2;
            #lineType = 2
            #for i2, point in enumerate(['A', 'B', 'C', 'D']):
            #    cv2.putText(rotatedBBOX_img, f"{point}{maskNumber}", box[i2], font, fontScale, fontColor, thickness,
            #                lineType)

            # we now need to warp the image (just rotation and cropping) to get only the cow region
            if self.printDebugInfoToScreen: print(f"\n\nbox points shape= {box.shape}")

            # directly from rect
            boxW, boxH = rect[1];
            boxW = int(boxW);
            boxH = int(boxH)

            # if boxAngle > 45:
            #    temp = boxW; boxW = boxH; boxH = temp #width and height need to be reversed

            cowArea = boxW * boxH
            cowAreaRatio = cowArea / (frame.shape[0] * frame.shape[1])
            cowAspRatio = boxW / boxH

            # remove later
            # cx, cy = rect[0]; bottomLeftCornerOfText = (int(cx), int(cy))
            # cv2.putText(rotatedBBOX_img, f'AR: {cowAspRatio:.3f}, ang={boxAngle:0.3f}', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if self.cowAspRatio_ulim is not None:

                if (cowAspRatio > 1 / self.cowAspRatio_ulim or cowAspRatio < self.cowAspRatio_ulim) and cowAreaRatio > self.cowAreaRatio_llim: #0.121:  # cowAR_ulim is 0.53, Aspect Ratio< 0.53 is for vertical cows, > 1/0.53 is for horizontal cows
                    # horizontal cow in barn cam
                    # allow this
                    countsDict['total_AR_area_pass'] += 1
                    pass

                else:
                    if self.printDebugInfoToScreen: print(f"skipping cow due to cow aspect ratio or area ratio limit")
                    if self.printDebugInfoToScreen: print(
                        f"current aspect ratio = {cowAspRatio}, upper limit = {self.cowAspRatio_ulim}")
                    logging.debug(
                        f"Skipping cow due to cow aspect ratio or area ratio limit. Current AR = {cowAspRatio}, upper lim AR = {self.cowAspRatio_ulim}.\nCurrent Area Ratio = {cowAreaRatio}, lower lim Area Ratio = {self.cowAreaRatio_llim}")

                    # put the cowID value in the center
                    text = f'AR: {cowAspRatio:.3f}, ang={boxAngle:0.3f}'
                    cx, cy = rect[0]
                    bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)

                    rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30),
                                                    (int(cx) + 500, int(cy) + 15), (0, 0, 0),
                                                    -1)  # background rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX;
                    fontScale = 1;
                    fontColor = (255, 255, 255);
                    thickness = 2;
                    lineType = 2
                    cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                                lineType)

                    continue

            if self.printDebugInfoToScreen: print(f"box later = {box}")
            if self.printDebugInfoToScreen: print(f"\nboxW = {boxW}, boxH = {boxH}\n")

            # now for KEYPOINTS
            instance_KPs = keypoints[maskNumber]  # keypoints of the current instance
            logging.debug(f"Original KPs = \n{instance_KPs}\n")

            #adding new code to force interpolation of center back keypoint - to reduce detection variance
            #pdb.set_trace()
            #logging.debug('Forcing center_back keypoint to become invisible (visibility = 0) so that it is always interpolated.') # This is to reduce variance in detection.
            #instance_KPs[self.kpn.index('center_back'),-1] = 0
            #pdb.set_trace(header='After forcing center_back visibility to 0.')


            # get visible keypoints
            visible_KPs, n_visibleKPs, totalKPs = get_visible_KPs(keypoints=instance_KPs,
                                                                  keypoint_names=self.keypoint_names)  # use later for interpolation
            nKPDetectionCounts[n_visibleKPs] += 1
            if n_visibleKPs == totalKPs:
                countsDict['total_allKps_det_inter'] += 1
            else:  # all 10 kps not visible - try to interpolate the missing kps
                instance_KPs = interpolateKPs(instance_KPs=instance_KPs, keypoint_names=self.keypoint_names,
                                              keypoint_threshold=0.5, frameW=frameW,
                                              frameH=frameH)  # using this to interpolate missing kps
                visible_KPs, new_n_visibleKPs, totalKPs = get_visible_KPs(keypoints=instance_KPs,
                                                                          keypoint_names=self.keypoint_names)
                if new_n_visibleKPs == totalKPs:
                    countsDict['total_allKps_det_inter'] += 1

            # check keypoints confidence
            cow_tv_KP_confidence, cow_tv_KP_maxConfidence = cow_tv_KP_rule_checker(keypoints=instance_KPs,
                                                                                   keypoint_names=self.keypoint_names)  # this is the original rule checker, checks rules with hand crafted thresholds
            # cow_tv_KP_confidence, cow_tv_KP_maxConfidence, instanceRulePassesDict = cow_tv_KP_rule_checker2(keypoints=instance_KPs, keypoint_names=self.keypoint_names, datasetKpStatsDict=self.datasetKpStatsDict, maskContour=cnt) #this rule checker uses dictionary of stat values, with limits computed from the dataset

            if self.printDebugInfoToScreen: print(
                f"cow_tv_KP_confidence = {cow_tv_KP_confidence}, cow_tv_KP_maxConfidence = {cow_tv_KP_maxConfidence}")

            if self.kpRules_maxPossibleConf is None and cow_tv_KP_maxConfidence != 0:  # so that it happens only once,
                self.kpRules_maxPossibleConf = cow_tv_KP_maxConfidence

                # kpRulesPassCounts = kpRulesPassCounts[:self.kpRules_maxPossibleConf + 1]  # trimming the histogram's X axis, +1 as count starts from 0 where 0 indicates that all 10 kps are not detected - this will not work here as the length of the list cannot be mutated from inside a function (IDK why). Only the values in the list can be changed.

            kpRulesPassCounts[cow_tv_KP_confidence] += 1  # updating the histogram

            rotatedBBOX_img_Safe = rotatedBBOX_img.copy()

            # if True:
            # draw KPs in the main frame - for all cows
            rotatedBBOX_img = draw_and_connect_keypoints(img=rotatedBBOX_img, keypoints=instance_KPs,
                                                         keypoint_threshold=0.5, keypoint_names=self.keypoint_names,
                                                         keypoint_connection_rules=self.keypoint_connection_rules)

            text = f'KP_conf: {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}'
            cx, cy = rect[0]
            bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)

            rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30),
                                            (int(cx) + 315, int(cy) + 15), (0, 0, 0), -1)  # background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX;
            fontScale = 1;
            fontColor = (0, 255, 255);
            thickness = 2;
            lineType = 2
            cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            kpCorrectionMethodUsed = None  # 'iter' or 'SBF' #to track which correction method has been used

            if cow_tv_KP_maxConfidence == 0:  # remove this inner redundant condition later
                if self.printDebugInfoToScreen: print(f"Skipping cow as not all keypoints are detected visible")
                if self.printDebugInfoToScreen: print(
                    f"cow tv KP confidence = {cow_tv_KP_confidence}, cow tv KP maxConfidence = {cow_tv_KP_maxConfidence}")
                logging.debug(
                    f"Skipping cow as not all keypoints are detected visible. Confidence = {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}")

                # break #remove break statement later
                continue  # must keep this!

            elif cow_tv_KP_confidence != cow_tv_KP_maxConfidence:  # We need (and cow_tv_KP_maxConfidence != 0). This is implied by the elif statement.

                if not self.keypointCorrectionOn:
                    if self.printDebugInfoToScreen: print(
                        f"Skipping cow instance. cow_tv_KP_confidence ({cow_tv_KP_confidence}) != cow_tv_KP_maxConfidence ({cow_tv_KP_maxConfidence}). Not attempting to correct keypoints as keypointCorrectionOn is {self.keypointCorrectionOn}")
                    logging.debug(
                        f"Skipping cow instance. cow_tv_KP_confidence ({cow_tv_KP_confidence}) != cow_tv_KP_maxConfidence ({cow_tv_KP_maxConfidence}). Not attempting to correct keypoints as keypointCorrectionOn is {self.keypointCorrectionOn}")
                    continue

                # try to correct misplaced KPs

                # try iterative method
                allRulesPassed, kpts, correctedKPs, RC1_Conf, RC2_Conf = correctMispalcedKPs_iterativeMode(
                    kpts=instance_KPs.copy(), maskContour=cnt, datasetKpStatsDict=self.datasetKpStatsDict,
                    affectedKPsDict=self.affectedKPsDict, kpWeightsDict=self.affectedKPsWeightsDict,
                    keypoint_names=self.kpn, keypoint_threshold=0.5, cowW=350, cowH=750, frameW=frameW, frameH=frameH)

                if allRulesPassed:

                    if self.useHandCraftedKPRuleLimits:
                        cow_tv_KP_confidence, cow_tv_KP_maxConfidence = RC1_Conf
                    else:
                        cow_tv_KP_confidence, cow_tv_KP_maxConfidence = RC2_Conf

                    instance_KPs = kpts
                    if self.printDebugInfoToScreen: print(
                        f"Cow keypoints errors eliminated by iterative keypoint correction. \nCow tv KP confidence = {cow_tv_KP_confidence}, cow tv KP maxConfidence = {cow_tv_KP_maxConfidence}")
                    logging.debug(
                        f"Cow keypoints errors eliminated by iterative keypoint correction. Confidence = {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}")

                    kpCorrectionMethodUsed = 'iter'

                else:  # try Strategic Brute Force
                    if self.printDebugInfoToScreen: print(
                        f"Iterative kp correction failed. RC1 confidence: {RC1_Conf} RC2_confidence: {RC2_Conf}")
                    logging.debug(
                        f"Iterative kp correction failed. RC1 confidence: {RC1_Conf} RC2_confidence: {RC2_Conf}")

                    allRulesPassed, kpts, correctedKPs, RC1_Conf, RC2_Conf = correctMispalcedKPs_StrategicBruteForceMode(
                        kpts=instance_KPs.copy(), maskContour=cnt, datasetKpStatsDict=self.datasetKpStatsDict,
                        affectedKPsDict=self.affectedKPsDict, kpWeightsDict=self.affectedKPsWeightsDict,
                        keypoint_names=self.kpn, keypoint_threshold=0.5, cowW=350, cowH=750, frameW=frameW,
                        frameH=frameH)

                    if allRulesPassed:

                        if self.useHandCraftedKPRuleLimits:
                            cow_tv_KP_confidence, cow_tv_KP_maxConfidence = RC1_Conf
                        else:
                            cow_tv_KP_confidence, cow_tv_KP_maxConfidence = RC2_Conf

                        instance_KPs = kpts
                        if self.printDebugInfoToScreen: print(
                            f"Cow keypoints errors eliminated by SBF keypoint correction. \nCow tv KP confidence = {cow_tv_KP_confidence}, cow tv KP maxConfidence = {cow_tv_KP_maxConfidence}")
                        logging.debug(
                            f"Cow keypoints errors eliminated by SBF keypoint correction. Confidence = {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}")

                        kpCorrectionMethodUsed = 'SBF'

                    else:
                        if self.printDebugInfoToScreen: print(
                            f"SBF kp correction failed. RC1 confidence: {RC1_Conf} RC2_confidence: {RC2_Conf}")
                        logging.debug(
                            f"SBF kp correction failed. RC1 confidence: {RC1_Conf} RC2_confidence: {RC2_Conf}")

                        if self.printDebugInfoToScreen: print(
                            f"\nSkipping cow instance due to low keypoint confidence. \nCow tv KP confidence = {cow_tv_KP_confidence}, cow tv KP maxConfidence = {cow_tv_KP_maxConfidence}")
                        logging.debug(
                            f"\nSkipping cow instance due to low keypoint confidence of = {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}")

                        continue

            # change KPs to corrected KPs in the image if KPs are corrected
            if kpCorrectionMethodUsed is not None:

                countsDict["total_kpCorrected_instances"] += 1

                if self.saveHardExamples:
                    # if self.printDebugInfoToScreen: print(f"KP Correction method used = {kpCorrectionMethodUsed}\nCorrectedKPs = {correctedKPs}")
                    # pdb.set_trace()

                    os.makedirs(f"{self.hardExamplesSavePath}/kpCorrectedImages/", exist_ok=True)
                    cv2.imwrite(
                        f'{self.hardExamplesSavePath}/kpCorrectedImages/{gt_label}_{frameCount}_{kpCorrectionMethodUsed}-Mode_before.jpg',
                        rotatedBBOX_img)  # just to visualize

                rotatedBBOX_img = draw_and_connect_keypoints(img=rotatedBBOX_img_Safe, keypoints=instance_KPs,
                                                             keypoint_threshold=0.5, keypoint_names=self.keypoint_names,
                                                             keypoint_connection_rules=self.keypoint_connection_rules)

                text = f'KP_conf: {cow_tv_KP_confidence}/{cow_tv_KP_maxConfidence}'
                cx, cy = rect[0]
                bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)

                rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30),
                                                (int(cx) + 315, int(cy) + 15), (0, 0, 0), -1)  # background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX;
                fontScale = 1;
                fontColor = (0, 255, 255);
                thickness = 2;
                lineType = 2
                cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                            lineType)

                if self.saveHardExamples:
                    os.makedirs(f"{self.hardExamplesSavePath}/kpCorrectedImages/", exist_ok=True)
                    cv2.imwrite(
                        f'{self.hardExamplesSavePath}/kpCorrectedImages/{gt_label}_{frameCount}_{kpCorrectionMethodUsed}-Mode_after.jpg',
                        rotatedBBOX_img)  # just to visualize
            # END OF KP CORRECTION CODE

            # SAVING EASY EXAMPLES
            if self.saveEasyExamples and kpCorrectionMethodUsed is None:
                # I know that I could have just put an else for the above block, but I just wanted these two blocks to be independent
                os.makedirs(f"{self.easyExamplesSavePath}/easyExampleImages/", exist_ok=True)
                cv2.imwrite(f'{self.easyExamplesSavePath}/easyExampleImages/{gt_label}_{frameCount}_withOverlay.jpg',
                            rotatedBBOX_img)  # just to visualize

            countsDict['total_allKPRulePass'] += 1

            # remove later
            # cv2.imshow('rotated_bbox_img', rotatedBBOX_img) #does not work :(
            # cv2.waitKey(0)

            # to be reconsidered
            # if boxAngle > 45:
            #    outBoxPts = np.array([[0,0], [boxW-1,0], [boxW-1, boxH-1], [0, boxH-1]])
            # elif boxAngle < 45:
            #    #input is ABCD but in this order
            #    '''
            #    B--C
            #    |  |
            #    A--D
            #    '''
            #    outBoxPts = np.array([[0, boxH - 1], [0, 0], [boxW - 1, 0], [boxW - 1, boxH - 1]])

            # use distance formula
            lenAB = int(np.sqrt((box[1, 0] - box[0, 0]) ** 2 + (box[1, 1] - box[0, 1]) ** 2))
            lenBC = int(np.sqrt((box[2, 0] - box[1, 0]) ** 2 + (box[2, 1] - box[1, 1]) ** 2))
            if lenAB < lenBC:
                boxW = int(lenAB);
                boxH = int(lenBC)  # smaller of the two sides must be the width remove this later if it does not work
                outBoxPts = np.array([[0, 0], [boxW - 1, 0], [boxW - 1, boxH - 1], [0, boxH - 1]])
            elif lenAB >= lenBC:
                boxW = int(lenBC);
                boxH = int(lenAB)  # smaller of the two sides must be the width remove this later if it does not work
                outBoxPts = np.array([[0, boxH - 1], [0, 0], [boxW - 1, 0], [boxW - 1, boxH - 1]])

            # Compute the perspective transform M
            M = cv2.getPerspectiveTransform(np.float32(box), np.float32(outBoxPts))  # homography matrix
            # cowRotatedCrop_img = cv2.warpPerspective(frame, M, (boxW, boxH), flags=cv2.INTER_LINEAR)
            cowRotatedCrop_img = cv2.warpPerspective(overlap, M, (boxW, boxH), flags=cv2.INTER_LINEAR)

            warpedKPs = np.zeros_like(instance_KPs)
            warpedKPs[:, :2] = cv2.perspectiveTransform(np.array([instance_KPs[:, :2]]).copy(),
                                                        M)  # the last column should have the probabilities (visibilities)
            warpedKPs[:, -1] = instance_KPs[:, -1]
            if self.printDebugInfoToScreen: print(f"Original KPs = \n{instance_KPs}\n\nWarpedKPs = {warpedKPs}")
            # logging.debug(f"Original KPs = \n{instance_KPs}\n\nWarpedKPs = {warpedKPs}\n")
            logging.debug(f"Warped KPs = \n{warpedKPs}\n")

            # align cow image to template
            # cowRotatedCropKPAligned_img, templateKPs = morph_cow_to_template(cowCropImg=cowRotatedCrop_img, warpedKPs=warpedKPs, keypoint_names=self.keypoint_names)
            cowRotatedCropKPAligned_img, templateKPs = morph_cow_to_template(cowCropImg=cowRotatedCrop_img,
                                                                             warpedKPs=warpedKPs,
                                                                             keypoint_names=self.keypoint_names,
                                                                             selectedTriIds=[3, 4, 5, 6, 7, 8, 9, 10,
                                                                                             11, 12, 13,
                                                                                             14])  # selecting only triagles inside the cow skeleton

            cropList.append(cowRotatedCropKPAligned_img.copy())


            text = f"GT: {gt_label}"
            cx, cy = rect[0]
            bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)

            rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30), (int(cx) + 315, int(cy) + 15),
                                            (0, 0, 0), -1)  # background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX;
            fontScale = 1;
            fontColor = (255, 255, 255);
            thickness = 2;
            lineType = 2
            cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if self.saveHardExamples and kpCorrectionMethodUsed is not None:
                # save image
                os.makedirs(f"{self.hardExamplesSavePath}/kpCorrectedImages/", exist_ok=True)
                imgName = f"{gt_label}_{frameCount}.jpg"
                cv2.imwrite(f'{self.hardExamplesSavePath}/kpCorrectedImages/{imgName}', frameSafe)  # just to visualize

                # as of now, there is only one cow per image in the training set. So, there will not be more than one entry per imgName in the hardExamplesDict.
                # But, I am writing the code this way to future-proof it - just in case we need to label multiple cows in the same frame, for example - in the barn video.
                dictEntry = {"gt_label": gt_label, "pred_label": predicted_cowID, "correctedKPs": correctedKPs,
                             "kpCorrectionMethodUsed": kpCorrectionMethodUsed, "instance_KPs": instance_KPs,
                             "instance_maskContours": cnt, "bbox": boxes[maskNumber], "frameW": frameW,
                             "frameH": frameH}
                if imgName not in self.hardExamplesDict:
                    self.hardExamplesDict[imgName] = [dictEntry]
                else:
                    self.hardExamplesDict[imgName].append(dictEntry)

            if self.saveEasyExamples and kpCorrectionMethodUsed is None:
                # save image
                os.makedirs(f"{self.easyExamplesSavePath}/easyExampleImages/", exist_ok=True)
                imgName = f"{gt_label}_{frameCount}.jpg"
                cv2.imwrite(f'{self.easyExamplesSavePath}/easyExampleImages/{imgName}', frameSafe)  # just to visualize

                # as of now, there is only one cow per image in the training set. So, there will not be more than one entry per imgName in the easyExamplesDict.
                # But, I am writing the code this way to future-proof it - just in case we need to label multiple cows in the same frame, for example - in the barn video.
                dictEntry = {"gt_label": gt_label, "pred_label": predicted_cowID, "instance_KPs": instance_KPs,
                             "instance_maskContours": cnt, "bbox": boxes[maskNumber], "frameW": frameW,
                             "frameH": frameH}
                if imgName not in self.easyExamplesDict:
                    self.easyExamplesDict[imgName] = [dictEntry]
                else:
                    self.easyExamplesDict[imgName].append(dictEntry)

            # COMPUTING THE BIT VECTORS
            # WE COMPUTE BIT VECOTRS ONLY FOR SIZE 16X16 for scaled image of size 512 X 1024 (8X8 for image size 256X512)
            threshVal = self.threshVal_CC if self.colorCorrectionOn else 127
            blkSize = 16
            gray = cv2.resize(cv2.cvtColor(cowRotatedCropKPAligned_img.copy(), cv2.COLOR_BGR2GRAY), (512, 1024))
            pixImg, fVec = pixelate(gray, blkSize=blkSize, type='avgThresh', threshVal=threshVal)  # stop at 16
            strVec = ''.join(str(x) for x in fVec)
            #bitVecStrList.append({f'blk{blkSize}':strVec}) #original
            bitVecStrList.append({'maskNumber':maskNumber,'bitVecStr':{f'blk{blkSize}':strVec}}) #to handle multiple cows per frame
            kpCorrectionMethodList.append({'maskNumber':maskNumber, 'kpCorrectionMethodUsed':kpCorrectionMethodUsed})  # to track which correction method has been used or if no correction method has been used



            # end of for loop

        cowAutoCatOutImg = None
        if not doNotComputeCowAutoCatOutImg:
            #While autoCattlogging from MultiInstances, we compute the cowAutoCatOutImg again after inserting the trackIDs in the rotatedBBOX_img. So, we can avoid double computation
            cowAutoCatOutImg = get_cow_autocattlog_frame(inFrame=rotatedBBOX_img, gtLabelsList=[gt_label], cropList=cropList, maxCows=3, frameNumber=frameCount)  # now takes images from top view KP aligned cattlog

        if self.printDebugInfoToScreen: print(f"GT Label = {gt_label}. Computed bit vector = {bitVecStrList}")
        logging.debug(f"GT Label = {gt_label}. Computed bit vector = {bitVecStrList}")




        #if gt_label is not None:  # use the predicted cow ID if you want - helpful in predicting on cut videos.
        #    return cowRecogEvalImg, predicted_cowID
        #else:
        #    return cowRecogEvalImg

        return bitVecStrList, kpCorrectionMethodList, cowAutoCatOutImg, cropList, rotatedBBOX_img #cowRotatedCropKPAligned_img


    def cattlogFromVideo_multiInstances(self, vidPath, cv2VideoWriterObj=None, frameFreq = 3, outRootDir="./autoCattlogMultiCow_outputs/", frame_width = None, frame_height = None, gt_label=None, countsDict=None, kpRulesPassCounts=None, kpRules_maxPossibleConf=None, nKPDetectionCounts=None, openTracks=[], closedTracks=[], startingTrackID=0, noDetFramesLimit=6, standalone=False, postProcessingFn=None):
        '''
        Track based cattlog generation.
        For AutoCattlog Version 2 (ACv2) - This along with its wrappers is what is used to get the results for the AutoCattlog Paper.
        
        Computes cattlog bitVector from any video - even uncut video.
        Can track multiple cows in the same frame.

        :param vidPath:
        :param frameFreq: picks one in every frameFreq frame for processing - proceeds to processing only if frameCount % frameFreq = 0 (this is not FPS)
        :param outRootDir:
        :param frame_width:
        :param frame_height:
        :param gt_label: the ground truth cowID
        :param openTracks: tracks that are open and will be updated. You can pass in the list of open tracks here.
        :param closedTracks: tracks that are closed and will not be updated anymore. This might not be necessary but it just makes coding easier. You can change this later.
        :param noDetFramesLimit: the number of contigious frames without any detections to wait before ending all the open tracks. We account for frameFreq below.
        :param standalone: If true, prints and logs stats of video file.
        :param postProcessingFn: optional post processing function to apply to the tracks after they are generated.

        :return:
        '''


        if standalone:

            os.makedirs(outRootDir, exist_ok=True)

            # need to create a new log file if these functions are called again from the same object and the caller requests a different filename
            # Remove all handlers associated with the root logger object.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=f'{outRootDir}/log_autoCattlog_OnVideoMultiCow_QR.log',
                                encoding='utf-8', level=logging.DEBUG, filemode='w',
                                format='%(asctime)s>> %(levelname)s : %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')  # set up logger

            logging.info(f"Running autoCattlog on a single video -- works with multiple instances of cows.")
            self.logInputParams()  # Saving all input params in the log file for - logging purposes ¯¯\__(O_O)__/¯¯

        logging.info(f"Processing video {vidPath}")
        logging.info( f"Input Params: cowAspRatio_ulim={self.cowAspRatio_ulim}, outDir = {outRootDir}")
        logging.info( f"Input Params: len(openTracks)={len(openTracks)}, len(closedTracks)={len(closedTracks)}, startingTrackID={startingTrackID}, noDetFramesLimit={noDetFramesLimit}")

        #input video details
        videoName = vidPath.split('/')[-1]

        #find file format
        fileFormat = videoName.split('.')[-1]
        logging.info(f"Input video format: {fileFormat}")

        #input video reader
        if fileFormat == 'hdf5' or fileFormat == 'hdf':
            cap = VideoCapture_HDF5(vidPath)
        else: #for .avi or other normal video formats
            cap = cv2.VideoCapture(vidPath)
    
        ret = True

        if frame_width == None or frame_height == None:
            # get width and height of video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.printDebugInfoToScreen:  print(f"frame_width = {frame_width}, frame_height = {frame_height}")
            logging.debug(f"frame_width = {frame_width}, frame_height = {frame_height}")
        
        # saving video
        if standalone:  # i.e. if no video writier object was supplied, create local video writer object
            #out = cv2.VideoWriter(f'{outRootDir}/autoCattlog_out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
            out = cv2.VideoWriter(f"{outRootDir}/autoCattlog_out_standalone_vid_{videoName.split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        else:
            #out = cv2.VideoWriter(f'{outRootDir}/autoCattlog_out_vid_{gt_label}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
            #out = cv2.VideoWriter(f'{outRootDir}/autoCattlog_out_vid_{gt_label}.mkv', -1, 30, (frame_width, frame_height))
            out = cv2.VideoWriter(f"{outRootDir}/autoCattlog_out_vid_{videoName.split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height)) #codec_id 001B

        
        if standalone: print(f"\n\nProcessing video {vidPath.split('/')[-1]}\n\n")

        

        frameCount = 0

        # inference stats setup for standalone mode
        if countsDict is None:
            countsDict = {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox': 0, 'total_instanceBBoxes': 0,
                          'total_AR_area_pass': 0, 'total_allKps_det_inter': 0, 'total_allKPRulePass': 0,
                          'total_correct_preds': 0, 'total_kpCorrected_instances': 0,
                          'gt_locs_list': []}  # change later ot total_correctPredictions if possible
        if kpRulesPassCounts is None:
            kpRulesPassCounts = np.array([0] * 100, int)  # to hold a histogram of number of KP rules passed
        # if kpRules_maxPossibleConf is None: #redundant
        #    kpRules_maxPossibleConf = None
        if nKPDetectionCounts is None:
            nKPDetectionCounts = np.array([0] * (len(self.keypoint_names) + 1), int)  # histogram of number of keypoints detected per image, +1 to include 0 - no visible kp case

        cow_bitVecStrList = [] #list of all bit vectors for a given cow from all frames of the video


        #TRACKS:
        trackId = startingTrackID #trackId for the next track to be created
        
        #to enable quick ending of tracks when there are no detections
        noDetFramesLimit = noDetFramesLimit//frameFreq #to account for frames skipped
        noDetFramesCounter = 0
        

        while ret:
            ret, frame = cap.read()
            if ret == False:
                if self.printDebugInfoToScreen: print(f"Breaking out since ret is false")
                break
            frameCount += 1

            #print(f"Current frame = {frameCount}")

            if self.printDebugInfoToScreen: print(f"Current frame = {frameCount}")
            # frameSafe = frame.copy()  # so that we do not write anything on this frame

            # remove later
            # if frameCount <129 or frameCount >318: #% 30 != 0:
            if frameCount % frameFreq != 0:
                if self.printDebugInfoToScreen: print(f"skipping frame")
                continue
            #if frameCount < 53000: #remove later
            #    continue
            # for debugging - remove later
            # if frameCount == 55710:
            #    cv2.imwrite(f"frameSample_{frameCount}.jpg", frame)
            # remove later
            #if frameCount > 60000:
            #   if self.printDebugInfoToScreen: print(f"breaking as per hard coding")
            #   break

            # logging.debug(f"\n\nProcessing Frame {frameCount}")
            countsDict['total_seenFrames'] += 1

            # if not self.useMMPOSE:
            #    img_prediction_op = self.predictOnImage_QR(frame=frame, countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, nKPDetectionCounts=nKPDetectionCounts, frameCount=frameCount, gt_label=gt_label)
            # else:

            cattlog_output = self.getBitVectorsFromFrame(frame=frame, countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, nKPDetectionCounts=nKPDetectionCounts, frameCount=frameCount, gt_label=None, doNotComputeCowAutoCatOutImg=True) #force gt_label to be None
            currentTrackPoints = [] #reinitialization

            if cattlog_output is not None: #it is none when there is now cow detected in the scene


                noDetFramesCounter = 0 #reset the noDetFramesCounter

                #bitVecStrList, kpCorrectionMethodList, cowAutoCatOutImg, cropList, rotatedBBOX_img  = cattlog_output
                bitVecStrList, kpCorrectionMethodList, _, cropList, rotatedBBOX_img  = cattlog_output #cowAutoCatOutImg will be None as we set doNotComputeCowAutoCatOutImg to True 

                #HERE WE WRITE THE LOGIC TO WORK ON MULTIPLE COWS IN THE SAME FRAME
                # WE NEED A MATCHING FUNCTION TO MATCH THE COWS IN THE CURRENT FRAME WITH THE COWS IN THE PREVIOUS FRAME

                #We have the prediction information, along with the rotated bbox locations for each cow in the frame.
                #We get these from the class variables: self.predictions, self.rotatedBboxPtsList
                #We need to create a current track point and pass it to the matching function.

                
                for i, rotatedBboxPts in enumerate(self.rotatedBboxPtsList):
                    currentTrackPoints.append(getNewTrackPoint(videoName=vidPath.split('/')[-1], frameNumber=frameCount, rotatedBBOX_locs=rotatedBboxPts))


                #add the bitvector to the track points
                if len(bitVecStrList) != 0: #len(bitVecStrList) = 0 when we don't get a complete set of 10 keypoints passing all rules even after keypoint rectification
                    for i, bitVecStr in enumerate(bitVecStrList):
                        #not all currentTrackPoints will have a bitVecStr generated. Only the ones that have a cow detected in them. Other track pts will just have rotated_bbox pts.
                        maskNum = bitVecStr['maskNumber'] 
                        currentTrackPoints[maskNum]['bitVecStr'] = bitVecStr['bitVecStr']['blk16'] 
   
                        #bitVecStrList and kpCorrectionMethod list are of the same lengths.                     
                        currentTrackPoints[maskNum]['kpCorrectionMethodUsed'] = kpCorrectionMethodList[i]['kpCorrectionMethodUsed'] #this is the method used to correct the keypoints for this cow. It will be None if no KPC was necessary.

                        #in its current form, kpCorrectionMethodUsed will be '' if not all keypoints are correctly detected.
                        # kpCorrectionMethodUsed will be None if no correction method was used but all keypoints are correctly detected.
                        # kpCorrectionMethodUsed will be iter if iter mode was used to correct the keypoints, and all keypoints are correct after rectification.
                        # kpCorrectionMethodUsed will be SBF if SBF mod was used to correct the keypoints, and all keypoints are correct after rectification.

                        #the same mask Number can be used to get the crop image for the cow
                        #sampleCropImg = cropList[i] #we do not care about the display order here. 

                #MATCH THE TRACK POINTS
                openTracks, closedTracks, trackId = matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount=frameCount)

                #Do some operations on the open tracks.
                for track in openTracks:
                    
                    #save the sample crop image for the track - the image of the first seen instance of the cow
                    if track['sampleCropImg'] is None and track['hasCow']: #these conditions select the first instnace of the cow when it has all keypoints correctly detected
                        #the latest trackPoint in the track will definitely have the bitVecStr as we are checking for the 'hasCow' condition along with the condition that the sampleCropImg is None
                        #If a cow has been found in the track, and the later bboxes do not have all keypoints, the sampleCropImg is not None. So this block is not entered.
                        #if you want to save the last image of the cow, then we need to keep overwriting the image on every find.
                        #for that use the condition - if track['trackPoints'][-1]['bitVecStr'] != '': instead.
                        try:
                            #the cropList and bitVecStrList are in the same order. Use this association to find out which sampleCropImage belongs to which track.
                            bitVecs_simplerList = [x['bitVecStr']['blk16'] for x in bitVecStrList] #list without the maskNumber Info
                            track['sampleCropImg'] = cropList[bitVecs_simplerList.index(track['trackPoints'][-1]['bitVecStr'])] 
                        except:
                            pdb.set_trace()
                                        
                    #display trackID on the track
                    trackPt = track['trackPoints'][-1]

                    #if a cow has been found in the track, start drawing a green bounding box around the track point instead of a red one
                    if track['hasCow']: rotatedBBOX_img = cv2.drawContours(rotatedBBOX_img,  [trackPt['rotatedBBOX_locs']], 0, (0, 252, 185), 3)

                    #DISPLAY THE TRACK ID
                    text = f"TrackID:{track['trackId']}"
                    cx, cy = trackPt['rotatedBBOX_locs'][0]
                    bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)
                    cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 2)

                    


                self.rotatedBboxPtsList = [] #[[0,0], [0,0], [0,0], [0,0]] #None #resetting the bbox points for the next frame
                self.predictions = None #resetting the predictions for the next frame


                #cowAutoCatOutImg = get_cow_autocattlog_frame(inFrame=rotatedBBOX_img, gtLabelsList=[gt_label], cropList=cropList, maxCows=3, frameNumber=frameCount)  # now takes images from top view KP aligned cattlog
                cowAutoCatOutImg = get_cow_autocattlog_frame(inFrame=rotatedBBOX_img, gtLabelsList=[gt_label]*len(cropList), cropList=cropList, maxCows=3, frameNumber=frameCount)  # now takes images from top view KP aligned cattlog

                # it will be none if no instances are found in it (i am doing this to reduce the output file size - change it to send a dummy frame without predictions if you want to keep the frame)
                out.write(cv2.resize(cowAutoCatOutImg, (frame_width, frame_height)))

                #cv2.imwrite(f"{outRootDir}/frame_{frameCount}.jpg", rotatedBBOX_img)

                #cv2.imshow('Rotated bbox img', rotatedBBOX_img)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #   if self.printDebugInfoToScreen: print(f"manual termination")
                #   break

            else:
                #if no cow is detected in the frame, we will close all open tracks after a certain number of frames
                #this allows us to handle accidently missed cows in the frame
                #the matching function will take care of closing the tracks

                if len(openTracks) != 0:

                    noDetFramesCounter += 1
                    logging.debug(f"No detections in frame {frameCount}. noDetFramesCounter = {noDetFramesCounter}. Adjusted Limit = {noDetFramesLimit} frames.")

                    if noDetFramesCounter == noDetFramesLimit:

                        #close all open tracks
                        logging.debug(f"No detections frame limit reached. Closing all open tracks.")

                        openTracks, closedTracks, trackId = matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount=frameCount)
                        noDetFramesCounter = 0
                
                else:
                    noDetFramesCounter = 0 #reset the noDetFramesCounter
                    logging.debug(f"No detections in frame {frameCount}. No open tracks to close. Moving on.")
                


        cap.release()
        out.release()

        
        
        ##SAVING THE CROPPED IMAGE FOR CATTLOG ENTRY
        ## you can later write a fn to create a bit vec image from the mode bit vector.
        ## saving a representational image for now
        #autoCattlog_imagesDir = f"./{outRootDir}/autoCattlog_crops{'_CC' if self.colorCorrectionOn else ''}"
        #os.makedirs(autoCattlog_imagesDir, exist_ok=True)
        #cv2.imwrite(f"{autoCattlog_imagesDir}/{gt_label}.jpg", sampleCropImg)



        if standalone:
            if self.kpRules_maxPossibleConf is not None: kpRulesPassCounts = kpRulesPassCounts[:self.kpRules_maxPossibleConf + 1]  # trimming the histogram's X axis, +1 as count starts from 0 where 0 indicates that all 10 kps are not detected
            log_cow_det_reid_stat(countsDict.copy(), kpRulesPassCounts, nKPDetectionCounts)
            
            
            #close all open tracks
            logging.info(f"Closing all open tracks. There are {len(openTracks)} open tracks.")

            for track in openTracks:
                track['endFrame'] = track['trackPoints'][-1]['frameNumber']
                closedTracks.append(track)

            openTracks = [] #reset the open tracks


            #SAVE SAMPLE CROP IMAGES
            #Saving all crops - to just see which all cows have appeared
            os.makedirs(f"{outRootDir}/autoCattlog_sampleCrops_fromAllClosedTracks/", exist_ok=True)
            for track in closedTracks:
                if track['hasCow']:
                    cv2.imwrite(f"{outRootDir}/autoCattlog_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])

            # POST PROCESS AND SAVE REQUIRED TRACKS AND BIT VECTORS
            if postProcessingFn is not None:
                #PROCESSING THE BIT VECTORS
                requiredTracks, bitVecCattlogDict = postProcessingFn(closedTracks, frameW=frame_width, printDebugInfoToScreen=True)     

                #save sample crop images only for required cows
                os.makedirs(f"{outRootDir}/autoCattlog_sampleCrops_fromRequiredTracks/", exist_ok=True)
                for track in requiredTracks:
                    cv2.imwrite(f"{outRootDir}/autoCattlog_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])
                    del track['sampleCropImg'] #deleting to save space  
                
                #SAVE THE NECESSARY FILES
                pickle.dump(requiredTracks, open(f"{outRootDir}/requiredTracks.pkl", 'wb'))
                pickle.dump(bitVecCattlogDict, open(f"{outRootDir}/cowDataMatDict_autoCattlogMultiCow_blk16{'_CC' if self.colorCorrectionOn else ''}.p", 'wb'))


            # SAVE EVERYTHING ELSE

            #delete the sampleCropImg key from the dictionary to save space
            for track in closedTracks:
                if 'sampleCropImg' in track: #the requiredTracks would have saved tracks by reference. So, some tracks might already have the sampleCropImg deleted from above.
                    del track['sampleCropImg']

            #SAVE THE NECESSARY FILES
            pickle.dump(openTracks, open(f"{outRootDir}/openTracks.pkl", 'wb')) #this should be empty
            pickle.dump(closedTracks, open(f"{outRootDir}/closedTracks.pkl", 'wb'))

        return openTracks, closedTracks, trackId


    def cattlogFromVideosList_multiInstances(self, vidPathsList, gt_labels_list=None, frameFreq=3, videosOutDir = "./autoCattlog_fromVideoList_multiInstances_outputs/", frame_width = None, frame_height = None, logDir=None, logSuffix='', postProcessingFn=postProcessTracks1):
        '''
        Autocattlog 2.0 / AutoCattlogV2
        Takes in list of unsegmented videos (hour-long videos) and proceses them to create a cattlog for each cow in each video.

        I call a separate post-processing function to create the final cattlog bit vectors. 
        In that function, I can fiilter out cows that walk in the opposite direction. 
        I can combine the GT labels with the track IDs in another function.

        :param vidPathsList: list of paths to video files (*.avi)
        :param gt_labels_list: UNUSED. I'm saving the tracks with trackIDs as the index. The GTlabels attached to the tracks using attachGTLabels in helpers.helper_for_autoCattlog.py module which can be used to process the tracks saved by this function.
                                    (list of GT labels (cowIDs) in the order in which the cows appear in the videos.)
        :param frameFreq: picks one in every frameFreq frame for processing - proceeds to processing only if frameCount % frameFreq = 0 (this is not FPS)
        :param videosOutDir: directory to save the output video files
        :param logDir: directory to save the log files
        :param logSuffix: suffix to add to the log file name
        :param postProcessingFn: optional post processing function to apply to the tracks after they are generated.

        :return: None
        
        :saved files in logDir:
        the log file
        openTracks: list of open tracks
        closedTracks: list of closed tracks
        requiredTracks: list of required tracks (the tracks after post-processing i.e. filtering out unnecessary tracks of cows moving in the wrong directions)
        bitVecCattlogDict: dictionary of bit vectors (serialized cow barcodes) for each cow in each video. The keys are the GT labels.
        Sample cropped, template aligned images: in {logDir}/autoCattlog_sampleCrops (Only for cows in required tracks. - Change it to save samples for all cows found if it can help you in cleaning data from a new dataset - to make one-to-one matches with the video output and the CSV file in the dairy or whatever.)
        '''

        bitVecCattlogDict = {} #we are only processing blkSize 16 as of now

        #create videosOutDir if it doesn't exist
        os.makedirs(videosOutDir, exist_ok=True)

        if logDir is None:
            logDir = videosOutDir #just to keep them all in one place

        # need to create a new log file if these functions are called again from the same object and the caller requests a different filename
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f'{logDir}/log_autoCattlog_OnVideoSet_multiInstances_QR{logSuffix}.log', encoding='utf-8', level=logging.DEBUG, filemode='w', format='%(asctime)s>> %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')  # set up logger

        logging.info(f"Creating cattlog (multi-instances) on video-list.")
        self.logInputParams() #Saving all input params in the log file for - logging purposes ¯¯\__(O_O)__/¯¯

        logging.info(f"Processing video list")

        # saving video
        #out = cv2.VideoWriter('autocattlog_vidList_multiInstnaces_out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        # inference stats setup
        countsDict = {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox': 0, 'total_instanceBBoxes': 0, 'total_AR_area_pass': 0, 'total_allKps_det_inter': 0, 'total_allKPRulePass': 0, 'total_correct_preds': 0, 'total_kpCorrected_instances':0, 'gt_locs_list': [], 'total_correct_pred_vids':0, 'topK_VideoLevelDict':{}, 'nVideos':len(vidPathsList), 'cowsWithNoCorrectPreds_list':[]}
        kpRulesPassCounts = np.array([0] * 100, int)  # to hold a histogram of number of KP rules passed
        kpRules_maxPossibleConf = None
        nKPDetectionCounts = np.array([0] * (len(self.keypoint_names) + 1), int)  # histogram of number of keypoints detected per image, +1 to include 0 - no visible kp case

        frameCount = 0

        #for idx, vidPath in enumerate(vidPathsList): #original

        pbar = tqdm(vidPathsList, unit='hrLongVideo')# tqdm progress bar
        openTracks, closedTracks = [], [] #initialize the tracks
        trackID = 0

        #original
        for idx, vidPath in enumerate(pbar): #you enumerate the pbar. if you tqdm(enumerate(vidList)), tqdm will not know how many cows are in the list, so it won't display eta or the progress bar!

            #gt_label = gt_labels_list[idx]
            
            #pbar.set_description(f"Cattlogging cow {gt_label}") #update progressbar description
            videoName = vidPath.split('/')[-1]
            pbar.set_description(f"Processing video {videoName}") #update progressbar description

            #countsDict and other such variables are passed by reference
            #cowBitVecStr = self.cattlogFromCutVideo(vidPath=vidPath, cv2VideoWriterObj=out, frameFreq=frameFreq, outRootDir=videosOutDir, frame_width=1920, frame_height=1080, gt_label=gt_label, countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, nKPDetectionCounts=nKPDetectionCounts, standalone=False)
            #bitVecCattlogDict[gt_label] = {'blk16':cowBitVecStr}

            #openTracks, closedTracks, trackID = self.cattlogFromVideo_multiInstances(vidPath, cv2VideoWriterObj=None, frameFreq = frameFreq, outRootDir=videosOutDir, frame_width = frame_width, frame_height = frame_height, gt_label=videoName.split('.')[0], countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, nKPDetectionCounts=nKPDetectionCounts, openTracks=openTracks, closedTracks=closedTracks, startingTrackID=trackID, noDetFramesLimit=6, standalone=False)
            openTracks, closedTracks, trackID = self.cattlogFromVideo_multiInstances(vidPath, cv2VideoWriterObj=None, frameFreq = frameFreq, outRootDir=videosOutDir, frame_width = frame_width, frame_height = frame_height, gt_label=None, countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, nKPDetectionCounts=nKPDetectionCounts, openTracks=openTracks, closedTracks=closedTracks, startingTrackID=trackID, noDetFramesLimit=6, standalone=False)


        # Using Joblib for parallel evaluation
        # Do not use this. Parallel evaluation messes up the logs. There will be nothing useful in the logfile. However, the evaluated videos turned out to be fine - so you could use this to get all evaluated cut-videos with overlays if you are in a hurry as this definitely runs faster than serial evaluation with the for loop.
        #_ = Parallel(n_jobs=8)(delayed(self.predictOnVideo2_QR)(vidPath=vidPath, cv2VideoWriterObj=None, frameFreq=frameFreq, outRootDir=videosOutDir, frame_width=1920, frame_height=1080, gt_label=gt_labels_list[idx], countsDict=countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, nKPDetectionCounts=nKPDetectionCounts, standalone=False) for idx, vidPath in enumerate(pbar)) # messes up the logs

        #out.release()

        
        #*************************************************************************************************************************#
        #LEGACY OVERHEADS
        if True: #just to allow me to collapse the code block
            total_correct_pred_vids = countsDict.pop('total_correct_pred_vids') #must remove this value before passing to log_cow_det_reid_stat function
            if self.kpRules_maxPossibleConf is not None: kpRulesPassCounts = kpRulesPassCounts[:self.kpRules_maxPossibleConf + 1]  # trimming the histogram's X axis, +1 as count starts from 0 where 0 indicates that all 10 kps are not detected
            log_cow_det_reid_stat(countsDict.copy(), kpRulesPassCounts, nKPDetectionCounts)
            pickle.dump({'countsDict': countsDict.copy(), 'kpRulesPassCounts': kpRulesPassCounts, 'nKPDetectionCounts' : nKPDetectionCounts}, open(f'{logDir}/metricsDict_autocattlog_onVideoSet_QR{logSuffix}.p','wb'))

            if self.saveHardExamples:
                os.makedirs(self.hardExamplesSavePath, exist_ok=True)
                pickle.dump(self.hardExamplesDict, open(f"{self.hardExamplesSavePath}/hardExamplesInfo.p", "wb"))

            if self.saveEasyExamples:
                os.makedirs(self.easyExamplesSavePath, exist_ok=True)
                pickle.dump(self.easyExamplesDict, open(f"{self.easyExamplesSavePath}/easyExamplesInfo.p", "wb"))
        #*************************************************************************************************************************#



        #POST Processing
        #close all open tracks
        logging.info(f"Closing all open tracks. There are {len(openTracks)} open tracks.")

        for track in openTracks:
            track['endFrame'] = track['trackPoints'][-1]['frameNumber']
            closedTracks.append(track)

        openTracks = [] #reset the open tracks

        #pass closed tracks to post porcessing fn - to remove cows walking in the opposite direction
        #compute bit vectors and save them
        #create association between available bit vectors and GT labels

        #SAVE SAMPLE CROP IMAGES
        #Saving all crops - to just see which all cows have appeared
        os.makedirs(f"{videosOutDir}/autoCattlog_sampleCrops_fromAllClosedTracks/", exist_ok=True)
        for track in closedTracks:
            if track['hasCow']: #otherwise the image will be None
                cv2.imwrite(f"{videosOutDir}/autoCattlog_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])

        # POST PROCESS AND SAVE BIT VECTORS IF A POST PROCESSING FN IS PROVIDED
        if postProcessingFn is not None:

            #PROCESSING THE BIT VECTORS
            requiredTracks, bitVecCattlogDict = postProcessingFn(closedTracks, frameW=frame_width, printDebugInfoToScreen=True)    

            #save sample crop images only for required cows
            os.makedirs(f"{videosOutDir}/autoCattlog_sampleCrops_fromRequiredTracks/", exist_ok=True)
            for track in requiredTracks:
                cv2.imwrite(f"{videosOutDir}/autoCattlog_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])
                del track['sampleCropImg'] #deleting to save space   
            
            pickle.dump(requiredTracks, open(f"{logDir}/requiredTracks.pkl", 'wb'))
            pickle.dump(bitVecCattlogDict, open(f"{logDir}/cowDataMatDict_autoCattlogMultiCow_blk16{'_CC' if self.colorCorrectionOn else ''}.p", 'wb'))

        ############################################################
        #SAVE EVERYTHING ELSE NOW
        
        #delete the sampleCropImg key from the dictionary to save space
        for track in closedTracks:
            if 'sampleCropImg' in track: #the requiredTracks would have saved tracks by reference. So, some tracks might already have the sampleCropImg deleted from above.
                del track['sampleCropImg']

        #SAVE THE NECESSARY FILES
        pickle.dump(openTracks, open(f"{logDir}/openTracks.pkl", 'wb')) #should be useless now
        pickle.dump(closedTracks, open(f"{logDir}/closedTracks.pkl", 'wb'))
        


def build_AutoCattlogger_from_config(ac_config=None, configPath=None):
    '''
    Utility function to build AutoCattlogger object from a config file.
    :param configPath: path to the config file (YAML format)
    :return: AutoCattlogger object
    '''

    assert (ac_config is not None) or (configPath is not None), "Either ac_config or configPath must be provided."

    if ac_config is None:
        ac_config = yaml.safe_load(open(configPath, 'r'))

    detectron2_device = int(ac_config['detectron2']['device']) if torch.cuda.is_available() else 'cpu' #= 0 if only 1 gpu is available

    # get detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(ac_config['detectron2']['model_config_path'])
    cfg.MODEL.DEVICE = detectron2_device
    cfg.MODEL.WEIGHTS = ac_config['detectron2']['weights_path']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ac_config['detectron2']['MODEL_ROI_HEADS_SCORE_THRESH_TEST']

    #mmpose config
    useMMPOSE = ac_config['mmpose']['useMMPOSE'] #should be True
    mmpose_poseConfigPath = ac_config['mmpose']['mmpose_poseConfigPath']
    mmpose_modelWeightsPath = ac_config['mmpose']['mmpose_modelWeightsPath']
    mmpose_device = ac_config['mmpose']['device'] if torch.cuda.is_available() else 'cpu'

    # pdb.set_trace(header="Building AutoCattlogger object from config")

    # build AutoCattlogger object
    cattlogger = AutoCattloger(cfg=cfg,
                              useHandCraftedKPRuleLimits=ac_config['AutoCattlogger'].get('useHandCraftedKPRuleLimits', True),
                              affectedKPsDictPath=ac_config['AutoCattlogger']['affectedKPsDictPath'],
                              datasetKpStatsDictPath=ac_config['AutoCattlogger']['datasetKpStatsDictPath'],
                              cowAspRatio_ulim=ac_config['AutoCattlogger'].get('cowAspRatio_ulim', 0.53),
                              cowAreaRatio_llim=ac_config['AutoCattlogger'].get('cowAreaRatio_llim', 0.042),
                              saveHardExamples=ac_config['AutoCattlogger'].get('saveHardExamples', False),
                              saveEasyExamples=ac_config['AutoCattlogger'].get('saveEasyExamples', False),
                              printDebugInfoToScreen=ac_config['AutoCattlogger'].get('printDebugInfoToScreen', False),
                              keypointCorrectionOn=ac_config['AutoCattlogger'].get('keypointCorrectionOn', True),
                              colorCorrectionOn=ac_config['AutoCattlogger'].get('colorCorrectionOn', True),
                              blackPointImgPth=ac_config['AutoCattlogger'].get('blackPointImgPth', None),
                              useMMPOSE=useMMPOSE,
                              mmpose_poseConfigPath=mmpose_poseConfigPath,
                              externalCC_fn=None, #ac_config['AutoCattlogger']['externalCC_fn'], #force None for now
                              threshVal_CC=None if ac_config['AutoCattlogger'].get('threshVal_CC', "None") == "None" else int(ac_config['AutoCattlogger']['threshVal_CC']),
                              mmpose_modelWeightsPath=mmpose_modelWeightsPath,
                              mmpose_device=mmpose_device)


    return cattlogger



if __name__ == "__main__":

    detectron2_device = 0 if torch.cuda.is_available() else 'cpu' #= 0 if only 1 gpu is available
    print(f"detectron2 device = {detectron2_device}")

    # mask and keypoint detection
    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("../configs/detectron2_configs/COW-InstanceSegmentation/mask_rcnn_R_50_C4_cow_tv.yaml") #mask only config

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.DEVICE = detectron2_device  # = 0, 1, 2 or 3 or 'cpu'

    # models we use
    cfg.MODEL.WEIGHTS = "../models/detectron2_models/maskRCNN_kpDatasetv6_orbbec2025_model_final.pth" # Mask Only, Orbbec Cam 2025 - kp_dataset_v6

    # outRootDir = "../../Outputs/Cow_TV_Crops/"

    cowAspRatio_ulim = 0.53  # to get almost full body images, no images with just partial cow body visible - ulim = upper limit
    cowAreaRatio_llim = 0.042 #0.121 #to reject very small detections as they could not be cows #llim = lower limit #0.121 for all holding area videos #0.042 for OpenBarn2024

    # color correction - for cow lighting project
    colorCorrectionOn = False  # True #False
    blackPointImgPth = f"../models/dairyIllumination_models/extendedBlkPtImg_normConvGaussian_kernelSize-301_6116.bmp" #NCG - BlackMirror Light Probes Paper?

    #for external color correction
    # sys.path.insert(0, './colorCorrectionModules/')

    #our CC method
    externalCC_fn = None; threshVal_CC = None #the threshval is set automatically (hard coded value is applied) for our CC method


    # save hard examples to be used for augmenting the keypoint detector training set
    saveHardExamples = False  # False
    saveEasyExamples = False  # False #True
    keypointCorrectionOn = True  # False #True - always set this to True unless you are trying to replicate EI2023 results

    # select to use MMPOSE or not
    useMMPOSE = True  # False - False Till SURABHI and CowColorCorrection project
    mmpose_poseConfigPath = '../configs/mmpose_configs/trainer_config/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_cow-tv_orbbecCam2025.py'
    mmpose_modelWeightsPath = '../models/mmpose_models/epoch_210_kpDatasetV5_2024.pth' # Orbbec cam 2025 - kp_dataset_v6
    mmpose_device = 'cuda:0'  # 'cpu

    # Object Oriented Python
    cattloger = AutoCattloger(cfg=cfg,
                              useHandCraftedKPRuleLimits=True,
                              affectedKPsDictPath="../models/keypointStat_models/datasetKpStatsDict_kp_dataset_v4_train_affectedKPs.yml",
                              datasetKpStatsDictPath="../models/keypointStat_models/datasetKpStatsDict_kp_dataset_v4_train.p",
                              cowAspRatio_ulim=cowAspRatio_ulim, cowAreaRatio_llim=cowAreaRatio_llim,
                              saveHardExamples=saveHardExamples, saveEasyExamples=saveEasyExamples,
                              printDebugInfoToScreen=False, keypointCorrectionOn=keypointCorrectionOn,
                              colorCorrectionOn=colorCorrectionOn, blackPointImgPth=blackPointImgPth,
                              useMMPOSE=useMMPOSE, mmpose_poseConfigPath=mmpose_poseConfigPath,
                              externalCC_fn = externalCC_fn, threshVal_CC=threshVal_CC,
                              mmpose_modelWeightsPath=mmpose_modelWeightsPath, mmpose_device=mmpose_device)

    ####################################################################################################################

    # uncut_vid_paths_list = glob.glob(f"/home/manu/hdd_1/Manu/temp_cache/videos-holding-250825_30fps/cam23*.avi") # Ubiquity cam data - 25 Aug 2025
    uncut_vid_paths_list = glob.glob(f"/home/manu/NAS02/Manu/Fall21/Videos/Original_Videos/videos-holding-220608-for-cattlog_30fps/cam24*.avi") # Ubiquity cam data - 8 June 2022

    uncut_vid_paths_list.sort() #to process in order - otherwise the files might not be listed in the chronological order

    for idx, vid_path in enumerate(uncut_vid_paths_list):
        print(f"vid path {idx} = {vid_path}")

    frame_width = None; frame_height = None #The code automatically selects

    cattloger.cattlogFromVideosList_multiInstances(vidPathsList=uncut_vid_paths_list, videosOutDir = "../outputs/AC_outputs/", frameFreq=1, postProcessingFn=postProcessTracks1) #ubiquity cam 2025 data - cows go from left to right

    


