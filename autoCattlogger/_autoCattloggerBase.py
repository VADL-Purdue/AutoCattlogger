'''
Author: Manu Ramesh | VADL | Purdue ECE
This file has class and functions to infer the Cow ID given images/videos of a Cows.
'''

#from detectron2.data.datasets import register_coco_instances
#regiester dummy dataset
#register_coco_instances("cow_dummy_dataset", {}, "../../../../Data/instance_seg_dummy/annotations/instances_default.json", "../../../../Data/instance_seg_dummy/images")

#regiester cow top view version 1 train and test datasets
#register_coco_instances("cow_topView_dataset_v1_train", {}, "../../../Data/cow_topview_dataset_v1/annotations/train.json", "../../../Data/cow_topview_dataset_v1/images")
#register_coco_instances("cow_topView_dataset_v1_test", {}, "../../../Data/cow_topview_dataset_v1/annotations/test.json", "../../../Data/cow_topview_dataset_v1/images")

#SOURCE
#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF
import pickle, yaml

import sys

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import json, cv2
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor

frame_width = 2304 #int(cap.get(3))
frame_height = 1296 #int(cap.get(4))
#out = cv2.VideoWriter('infout.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

import pdb, glob

#mmpose
from mmpose.apis import init_model


sys.path.insert(0, '../') #top level dir
from autoCattlogger.helpers.helper_for_kpCorrection import computeWeightsForMisplacedKPs

import logging
import warnings
warnings.simplefilter('ignore', np.RankWarning) #ignoreing warnings here. Comment this line if you want to see the "RankWarning: Polyfit may be poorly conditioned" warning




#for finding cows with same bit vectors
from autoCattlogger.helpers.helper_for_QR_and_inference import findSameBitVecCows






class _AutoCattloggerBase():
    '''
    The template/base class for AutoCattlogger system and other supporting modules.:
    '''
    #def __init__(self, cfg, bitVectorCattlogPath="./Models/bit_vector_cattlogs/cowDataMatDict_KP_aligned_june_08_2022.p", cattlogDirPath = "../../Outputs/top_view_KP_aligned_cattlog_june_08_2022/", useHandCraftedKPRuleLimits = True, datasetKpStatsDictPath="./output/statOutputs/datasetKpStatsDict_kp_dataset_v4_train.p", affectedKPsDictPath = None, cowAspRatio_ulim = None, saveHardExamples=False, hardExamplesSavePath="./output/hardExamples/", saveEasyExamples=False, easyExamplesSavePath="./output/easyExamples/", printDebugInfoToScreen=False, keypointCorrectionOn=True, colorCorrectionOn=False, blackPointImgPth=None, useMMPOSE=False, mmpose_poseConfigPath=None, mmpose_modelWeightsPath=None, mmpose_device='cuda:0'):
    def __init__(self, cfg, bitVectorCattlogPath=None, cattlogDirPath="../../Outputs/top_view_KP_aligned_cattlog_june_08_2022/", useHandCraftedKPRuleLimits=True, datasetKpStatsDictPath="../models/keypointStat_models/datasetKpStatsDict_kp_dataset_v4_train.p", affectedKPsDictPath=None, cowAspRatio_ulim=None, cowAreaRatio_llim = 0.121, saveHardExamples=False, hardExamplesSavePath="./output/hardExamples/", saveEasyExamples=False, easyExamplesSavePath="./output/easyExamples/", printDebugInfoToScreen=False, keypointCorrectionOn=True, colorCorrectionOn=False, blackPointImgPth=None, threshVal_CC= 50, externalCC_fn=None, useMMPOSE=False, mmpose_poseConfigPath=None, mmpose_modelWeightsPath=None, mmpose_device='cuda:0', discountSameBitVecCows=True):
        self.cfg = cfg

        self.cowAspRatio_ulim = cowAspRatio_ulim
        self.cowAreaRatio_llim = cowAreaRatio_llim #lower limit for cow area ratio. Cow area ratio = (area of cow bbox)/(area of video frame). Set a lower limit if you want to detect smaller cows.

        self.cattlogDirPath = cattlogDirPath #directory containing cattlog images in the format <cowID>.jpg

        self.keypoint_names = self.kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]
        self.keypoint_connection_rules = [(self.kpn[1 - 1], self.kpn[4 - 1], (0, 0, 255)), (self.kpn[1 - 1], self.kpn[5 - 1], (0, 255, 0)), (self.kpn[1 - 1], self.kpn[2 - 1], (255, 0, 0)), (self.kpn[2 - 1], self.kpn[3 - 1], (255, 255, 0)), (self.kpn[2 - 1], self.kpn[4 - 1], (0, 255, 255)), (self.kpn[3 - 1], self.kpn[4 - 1], (0, 0, 0)), (self.kpn[3 - 1], self.kpn[7 - 1], (255, 255, 255)), (self.kpn[4 - 1], self.kpn[6 - 1], (255, 128, 128)), (self.kpn[4 - 1], self.kpn[5 - 1], (128, 255, 128)), (self.kpn[4 - 1], self.kpn[7 - 1], (128, 128, 255)), (self.kpn[5 - 1], self.kpn[6 - 1], (255, 255, 128)), (self.kpn[5 - 1], self.kpn[8 - 1], (255, 128, 255)), (self.kpn[6 - 1], self.kpn[7 - 1], (128, 255, 255)), (self.kpn[6 - 1], self.kpn[8 - 1], (255, 128, 64)), (self.kpn[6 - 1], self.kpn[10 - 1], (255, 64, 128)), (self.kpn[6 - 1], self.kpn[9 - 1], (128, 255, 64)), (self.kpn[7 - 1], self.kpn[10 - 1], (128, 64, 255)), (self.kpn[8 - 1], self.kpn[9 - 1], (64, 255, 128)), (self.kpn[9 - 1], self.kpn[10 - 1], (64, 128, 255))]

        self.predictor = DefaultPredictor(self.cfg) #keypoint and mask predictors from Detectron2 - this might use GPU0 only by default.

        if bitVectorCattlogPath is not None: #it will be None by default now. We don't need a value for autocattlog functions - which are under child class that calls also call this init fn.
            self.bitVectorCattlogPath = bitVectorCattlogPath
            self.bitVectorCattlog = pickle.load(open(self.bitVectorCattlogPath, 'rb'))

            # Finding cows that share the same bit vector (blk16 - block size 16X16)
            self.sameBVecCows, self.sameBVecCowsDict = findSameBitVecCows(bitVecCattlog=self.bitVectorCattlog)
        else:
            self.bitVectorCattlogPath = None
            self.bitVectorCattlog = None

        self.K_valueForTopKProbing = 3 #the K Value for probing top-K predictions at the instance level

        self.useHandCraftedKPRuleLimits = useHandCraftedKPRuleLimits #use KP Rule limits from annotation data
        self.datasetKpStatsDictPath = datasetKpStatsDictPath
        self.datasetKpStatsDict = None

        #if not self.useHandCraftedKPRuleLimits:
        # always set this variable irrespective of if useHandCraftedKPRuleLimits is set or not, as, this datasetKpStatsDict will be used to correct misplaced KPs
        self.datasetKpStatsDict = pickle.load(open(self.datasetKpStatsDictPath, 'rb')) #the dictionary with max and min values for KP rule checker

        self.kpRules_maxPossibleConf = None #updated upon prediction of first instance. Does not change later.

        self.affectedKPsDict = yaml.safe_load(open(affectedKPsDictPath, 'r'))
        self.affectedKPsCountsDict, self.affectedKPsWeightsDict = computeWeightsForMisplacedKPs(affectedKPsDict=self.affectedKPsDict, keypoint_names=self.kpn)

        #for Surabhi Experiments Phase 3
        # hard examples are all instances where the model gets the keypoint predictions wrong, but are successfully corrected by the keypoint correction methodology
        self.saveHardExamples = saveHardExamples #if True, will save images and annotations of those examples where keypoint detector failed but the keypoint correction algorithm was successful.
        self.hardExamplesDict = {}
        self.hardExamplesSavePath = hardExamplesSavePath

        # For Surabhi Experiments Phase 4
        # easy examples are all examples where the model gets the keypoint predictions right, without the need for keypoint correction
        self.saveEasyExamples = saveEasyExamples
        self.easyExamplesDict = {}
        self.easyExamplesSavePath = easyExamplesSavePath

        self.printDebugInfoToScreen = printDebugInfoToScreen

        self.keypointCorrectionOn = keypointCorrectionOn

        #color correction - beyond SURABHI
        self.colorCorrectionOn = colorCorrectionOn #used to turn on black point correction for cow lighting project
        self.blackPointImgPth = blackPointImgPth #storing this value also, for logging purposes
        if blackPointImgPth is not None and colorCorrectionOn:
            self.blackPointImg = cv2.imread(blackPointImgPth) #image used for black point correction/color correction
        else:
            self.blackPointImg = None

        self.externalCC_fn = externalCC_fn #pass an external color correction function if you want to use that instead of ours
        if self.externalCC_fn is None:
            # 50 #The threshold value for binarizaing grayscale images when color correction is on. The default value of 127 is used when colorCorrection is off.
            self.threshVal_CC = 75 #127 #75 #100 #50 #hard coding implicit (our) method
        else:
            self.threshVal_CC = threshVal_CC


        #MMPOSE
        self.useMMPOSE = useMMPOSE #whether or not to use MMPOSE for keypoints
        if self.useMMPOSE:
            self.mmpose_poseConfigPath = mmpose_poseConfigPath
            self.mmpose_modelWeightsPath = mmpose_modelWeightsPath

            # build pose estimator
            self.mmpose_device = mmpose_device #'cuda:0'  # 'cpu
            self.pose_estimator = init_model(self.mmpose_poseConfigPath, self.mmpose_modelWeightsPath, device=self.mmpose_device)
            # cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

        #DISCOUNTING COWS WITH SAME BIT VECTORS
        # Cows can have the same bit vectors if they have uniform coat patterns - such as a completely black cow.
        self.discountSameBitVecCows = discountSameBitVecCows #if true, the system will treat predicted IDs that have the same bit vector as the ground truth, as correct predictions.



    def logInputParams(self):
        '''
        For logging important class params.
        This is useful when retracing runs from the log file.
        Also helps in identifying log files in case of mis-namings.
        :return:
        '''


        paramsForLog = self.__dict__.copy()
        del paramsForLog['cfg'], paramsForLog['kpn'], paramsForLog['keypoint_connection_rules'], paramsForLog[
            'predictor'], paramsForLog['bitVectorCattlog'], paramsForLog['blackPointImg'], paramsForLog['externalCC_fn']

        #adding param
        paramsForLog['externalCC_fn'] = self.externalCC_fn.__name__ if self.externalCC_fn is not None else 'None'

        #modifying param
        paramsForLog['blackPointImgPth'] = self.blackPointImgPth if self.blackPointImgPth is not None else 'None' #it is None (or not used) even when color correction is set when we pass in external color correction function

        if self.bitVectorCattlogPath is not None:
            del paramsForLog['sameBVecCowsDict'] # we need only the list to be logged and not the dict. They have the same info.

        if self.useMMPOSE:
            del paramsForLog['pose_estimator'] # This is not serializable.

        #print(f"{paramsForLog.keys()}")
        #pdb.set_trace()

        logging.info(f"\n\n####################################################################################################################\n")
        logging.info(f"Important input parameters: \n{json.dumps(paramsForLog, indent=4, sort_keys=True)}")

        logging.info(f"\n\n####################################################################################################################\n")
        logging.info(f"Detectron2 Config: (Use this to verify what weights were used for evaluation.) \n{self.cfg.dump()}")
        logging.info(f"\n\n####################################################################################################################\n")

        return

if __name__ == "__main__":

    pass
