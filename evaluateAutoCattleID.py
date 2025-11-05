'''
Author: Manu Ramesh

This script has code to evaluate the AutoCattlog bitvectors and generate the results for the AutoCattlogger paper.

'''
import cv2, pickle, pdb, glob, os, numpy as np, sys, pdb

# sys.path.insert(0, '../')
from autoCattlogger.helpers.helper_for_QR_and_inference import inferBitVector, findSameBitVecCows
from autoCattlogger.helpers.helper_for_infer import printAndLog
from collections import Counter
from tqdm import tqdm, trange

import logging
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count
import multiprocessing
import pandas as pd
from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors

import matplotlib.pyplot as plt

from autoCattleID.autoCattleID import evalTrackBitVecs_multiProc
# from autoCattleID.helpers.helper_for_trackPointFiltering import trackPtsFilter_bySizeOfRotatedBBoxes, trackPtsFilter_byProximityOfRBboxToFrameCenter, trackPtsFilter_closestImgToFrameCenter, trackPtsFilter_ablationStudy_trainingDay, trackPtsFilter_ablationStudy_evalDay


from argparse import ArgumentParser


#############################################################################################################################
###################### FUNCTIONS TO RUN THE EVALUATIONS WITH DIFFERENT FILTER FUNCTIONS AND PARAMETERS ######################


def evaluateACID(cattlogDict_train=None, trainTracksPath='', evalTracksPath='', outDir='../outputs/', logSuffix='', filterCattlogTracks=True, filterEvalTracks=False, trackPtsFilterFnName=None, discountSameBitVecCows=True, excludeIDsCSVPath=None, **kwargs):
    '''
    Wrapper to evaluate AutoCattlog bit vectors (barcodes) for cow identification.

    :param trainTracksPath: Path to the training tracks list pickle file.  This file is output by the AutoCattlogger.
    :param evalTracksPath: Path to the evaluation tracks list pickle file. This file is output by the AutoCattlogger.
    :param outDir: directory to save the log files and other outputs.
    :param logSuffix: suffix to add to the log file name.
    :param filterCattlogTracks: If True, the trackpoints in the training tracks are filtered using the trackPtsFilterFnName and a new autoCattlog bitvector is generated for each cow.
    :param filterEvalTracks: If True, the trackpoints from the eval tracks are filtered using the trackPtsFilterFnName.
    :param trackPtsFilterFnName: function name to filter the trackpoints in the training and eval day tracks. This function should take in a list of trackpoints and return a filtered list of trackpoints. The function should also take in any additional arguments that are passed to it.
    :param discountSameBitVecCows: If True, the cows with same bit vectors are discounted. Discounting means to consider predictions of a different cowID with the same training bit vector as correct predictions. (Refer BMLP paper for more details.)
    :param excludeIDsCSVPath: Path to a CSV file with a column named "excludeIDs" that lists the cow IDs to be excluded from evaluation.
    :param **kwargs: additional arguments that are passed to the trackPtsFilterFn

    :return: instanceAcc, nInstances, trackAcc, nCowTracks
    '''



    #Setup Logger
    #Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # logging.basicConfig(filename=f'{logDir}/log_trackLevelEval{logSuffix}.log', encoding='utf-8', level=logging.DEBUG, filemode='w', format='%(asctime)s>> %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')  # set up logger
    logging.basicConfig(filename=f'{outDir}/log_trackLevelEval{logSuffix}.log', encoding='utf-8', level=logging.INFO, filemode='w', format='%(asctime)s>> %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')  # set up logger
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) #to suppress matplotlib font manager warnings - otherwise it will add a lot of clutter to the log file


    printAndLog(f"Evaluating AutoCattlog bit vectors for cow identification.\n", printDebugInfoToScreen=True)
    # Print all parameters to screen and log file
    printAndLog(f"********************\nInput Parameters:\n********************\n", printDebugInfoToScreen=True)
    printAndLog(f"trainTracksPath = {trainTracksPath}", printDebugInfoToScreen=True)
    printAndLog(f"evalTracksPath = {evalTracksPath}", printDebugInfoToScreen=True)
    printAndLog(f"outDir = {outDir}", printDebugInfoToScreen=True)
    printAndLog(f"logSuffix = {logSuffix}", printDebugInfoToScreen=True)
    printAndLog(f"filterCattlogTracks = {filterCattlogTracks}", printDebugInfoToScreen=True)
    printAndLog(f"filterEvalTracks = {filterEvalTracks}", printDebugInfoToScreen=True)
    printAndLog(f"trackPtsFilterFnName = {trackPtsFilterFnName}", printDebugInfoToScreen=True)
    printAndLog(f"discountSameBitVecCows = {discountSameBitVecCows}", printDebugInfoToScreen=True)
    
    printAndLog(f"excludeIDsCSVPath = {excludeIDsCSVPath}", printDebugInfoToScreen=True)

    tracksList_train  = pickle.load(open(trainTracksPath, 'rb'))
    tracksList_eval   = pickle.load(open(evalTracksPath, 'rb'))

    # Delete all cow tracks with cowIDs in the excludeIDs list
    if excludeIDsCSVPath is not None:
        excludeIDsDF = pd.read_csv(excludeIDsCSVPath)
        excludeIDsList = excludeIDsDF['excludeIDs'].tolist()
        printAndLog(f"Excluding cow IDs: {excludeIDsList} from evaluation.", printDebugInfoToScreen=True)

        tracksList_eval = [track for track in tracksList_eval if track['gt_label'] not in excludeIDsList]



    instanceAcc, nInstances, trackAcc, nCowTracks = evalTrackBitVecs_multiProc(cattlogDict_train=cattlogDict_train, tracksList_train=tracksList_train, tracksList_eval=tracksList_eval, discountSameBitVecCows=discountSameBitVecCows, filterCattlogTracks=filterCattlogTracks, filterEvalTracks=filterEvalTracks, logDir=outDir, logSuffix=logSuffix, trackPtsFilterFnName=trackPtsFilterFnName, **kwargs)


if __name__ == '__main__':


    parser = ArgumentParser(description="Evaluate AutoCattlog bit vectors (barcodes) for cow identification.")

    ioGroup = parser.add_argument_group('IO Options')
    ioGroup.add_argument('-c', '--cattlogDictTrainPath', type=str, default=None, help='Path to the training cattlog dictionary pickle file with all the bit-vectors. If None, will be generated from the training tracks (after filtering if needed). You can use it if you already have the required bit-vectors saved.')
    ioGroup.add_argument('-t', '--trainTracksPath', type=str, required=True, help='Path to the training tracks list pickle file.')
    ioGroup.add_argument('-e', '--evalTracksPath', type=str, required=True, help='Path to the evaluation tracks list pickle file.')
    ioGroup.add_argument('-o', '--outDir', type=str, default='./', help='directory to save the log files and other outputs.')
    ioGroup.add_argument('-l', '--logSuffix', type=str, default='', help='suffix to add to the log file name.')


    
    filterGroup = parser.add_argument_group('TrackPoints Filtering Options')
    filterGroup.add_argument('--filterCattlogTracks', type=bool, default=True, help='If True, the trackpoints in the training tracks are filtered using the trackPtsFilterFn and a new autoCattlog bitvector is generated for each cow.')
    filterGroup.add_argument('--filterEvalTracks', type=bool, default=False, help='If True, the trackpoints from the eval tracks are filtered using the trackPtsFilterFn.')
    filterGroup.add_argument('--trackPtsFilterFnName', type=str, default=None, help='Name of the function to filter the trackpoints in the training and eval day tracks. Choose from available options. If None, no filtering is done.')

    otherGroup = parser.add_argument_group('Other Options')
    otherGroup.add_argument('-d', '--discountSameBitVecCows', type=bool, default=True, help='If True, the cows with same bit vectors are discounted. Discounting means to consider predictions of a different cowID with the same training bit vector as correct predictions. (Refer BMLP paper for more details.)')
    otherGroup.add_argument('--excludeIDsCSVPath', type=str, default=None, help='Path to a CSV file with a column named "excludeIDs" that lists the cow IDs to be excluded from evaluation.')

    args = parser.parse_args()

    evaluateACID(cattlogDict_train=args.cattlogDictTrainPath, trainTracksPath=args.trainTracksPath, evalTracksPath=args.evalTracksPath, outDir=args.outDir, logSuffix=args.logSuffix, filterCattlogTracks=args.filterCattlogTracks, filterEvalTracks=args.filterEvalTracks, trackPtsFilterFnName=args.trackPtsFilterFnName, discountSameBitVecCows=args.discountSameBitVecCows, excludeIDsCSVPath=args.excludeIDsCSVPath)