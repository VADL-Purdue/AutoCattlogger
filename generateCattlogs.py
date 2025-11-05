'''
Author: Manu Ramesh

Has functions to generate the cattlogs using autoCattlogger.
'''

import yaml
import cv2, pickle, pdb, glob, os, numpy as np
import sys, pandas as pd

# from autoCattlog import AutoCattloger, build_AutoCattlogger_from_config
# from autoCattlogger.autoCattlogger import build_AutoCattlogger_from_config
from autoCattlogger.autoCattlogger import build_AutoCattlogger_from_config
# import torch
# from detectron2.config import get_cfg

# from autoCattlogger.helpers.helper_for_autoCattlog import createCattlogImages_fromBitVectors, attachGTLabels, getCutsInfoList
from autoCattlogger.helpers.helper_for_autoCattlog import createCattlogImages_fromBitVectors, attachGTLabels, getCutsInfoList
import argparse, yaml

#from multiprocessing import Pool #does not work with cuda
#import torch.multiprocessing as mp


class CattlogGenerator():

    def __init__(self, configPath):

        #CREATE THE COMMON AUTOCATTLOGGER OBJECT
        self.ac_config = yaml.safe_load(open(configPath, 'r'))
        self.cattlogger = build_AutoCattlogger_from_config(ac_config=self.ac_config)

    def generateCattlogs(self, srcDir, outDir, frameFreq=1, gtLabelsCSV_path=None, forceRegenerate=False):
        '''
        Generates the cattlog
        '''
        videosOutDir = outDir
        # assert not os.path.exists(videosOutDir), f"Overwrite protection. Output directory {videosOutDir} already exists. Please delete it and run again."
        uncut_vid_paths_list = glob.glob(f"{srcDir}/*.avi") #Train day

        #Glob can return files in any order! So we must srot them.
        #sort in order of file name - the file name has start times. So, this should sort it in order of start times.
        uncut_vid_paths_list.sort()

        #uncut_vid_paths_list = uncut_vid_paths_list[2:] #NOTE: remove later, only for debugging

        if not os.path.exists(videosOutDir) or forceRegenerate:
            print(f"Generating cattlogs for {len(uncut_vid_paths_list)} videos from {srcDir} and saving to {videosOutDir}")
            self.cattlogger.cattlogFromVideosList_multiInstances(vidPathsList=uncut_vid_paths_list, videosOutDir=videosOutDir, frameFreq=frameFreq)
        
        else:
            print(f"\n***Output directory {videosOutDir} already exists. Skipping cattlog generation. Use command line option -f or --forceRegenerate to force regeneration.***")


        #ATTACH GT LABELS TO TRACKS
        if gtLabelsCSV_path is not None:
            print(f"Attaching GT labels to tracks, bitVectorCattlog, and sampleCropImages")

            tracksList = pickle.load(open(f'{videosOutDir}/requiredTracks.pkl', 'rb'))
            sampleCropImgsDir= f"{videosOutDir}/autoCattlog_sampleCrops" #must not have ending '/'
            bitVecCattlogPath = f"{videosOutDir}/cowDataMatDict_autoCattlogMultiCow_blk16{'_CC' if self.colorCorrectionOn else ''}.p"

            tracksWithGTLabels, bvCattlogWithGTLabels = attachGTLabels(gtLabelsCSV_path=gtLabelsCSV_path, tracksList=tracksList, saveDir=videosOutDir, sampleCropImgsDir=sampleCropImgsDir, bitVecCattlogPath=bitVecCattlogPath)
            

            #CREATE CATTLOG IMAGES FROM BIT VECTORS
            print(f"Creating cattlog images from bit vectors")
            createCattlogImages_fromBitVectors(bvCattlogWithGTLabels, outDir=f"{videosOutDir}/pixBinImages_withGTLabels{'_CC' if self.colorCorrectionOn else ''}")


            #GET AUTO CUTS INFO
            print(f"Getting autoCuts info")
            _ = getCutsInfoList(tracksWithGTLabels, saveDir=videosOutDir)
        
        else:
            print(f"No GT labels CSV path provided. Skipping the ground truth label attachment step.")


   

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='Generate cattlogs with the AutoCattlogger.')
    parser.add_argument('-c', '--ac_configPath', type=str, required=False, help='Path to the AutoCattlogger configuration YAML file.', default='../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml')
    parser.add_argument('-s', '--srcdir', type=str, required=True, help='Directory containing (only) the top-view videos to process.')
    parser.add_argument('-o', '--outdir', type=str, required=False, help='Output directory to save results.', default='./outputs/autoCattlogger_outputs/')
    parser.add_argument('-F', '--frameFreq', type=int, required=False, help='Frame frequency for processing videos. One in every frameFreq frames will be processed.', default=1)
    parser.add_argument('-g', '--gtlabelsCSV_path', type=str, required=False, help='Path to the CSV file containing ALIGNED ground truth labels for the videos.', default=None)
    parser.add_argument('-f', '--forceRegenerate', action='store_true', help='If set, will force regeneration of cattlogs even if output directory exists.')
    
    args = parser.parse_args()

    acGen = CattlogGenerator(configPath=args.ac_configPath)
    acGen.generateCattlogs(srcDir=args.srcdir, outDir=args.outdir, frameFreq=args.frameFreq, gtLabelsCSV_path=args.gtlabelsCSV_path, forceRegenerate=args.forceRegenerate)

