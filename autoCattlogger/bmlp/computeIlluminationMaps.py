'''
Author: Manu Ramesh

Has functions to compute  the illumination maps from tracks of black cows from each month of data.
This is for ACv2.
'''


import pickle, pdb, glob, os, sys, torch
from detectron2.config import get_cfg
import cv2 

sys.path.append('../../') #add top level dir to path

from autoCattlogger.bmlp.cowColorCorrector import build_CowColorCorrector_from_config, CowColorCorrector
import argparse

def computeIlluminationMaps(args):


    if args.mode == 'fullCompute':
        tracksListPath = args.tracksListPath
        inputVideoDir = args.inputVideoDir
        outputDir = args.outputDir
        cowID = args.cowID
        trackID = args.trackID
        useMultiProc = args.useMultiProc
        numWorkers = args.numWorkers
        configPath = args.configPath

        # load the tracks list
        tracksList = pickle.load(open(tracksListPath, 'rb'))

        cccObject = build_CowColorCorrector_from_config(configPath=configPath)

        # compute the scene illumination map
        if useMultiProc:
            completeBlkPtImg, completeWhtPtImg = cccObject.computePatchWiseBlackPointsFromTrack_multiProc(tracksList=tracksList, srcVideosDir=inputVideoDir, gt_label=cowID, trackId=trackID, outImgDir=outputDir, patchSize=1, filterFrames=True, computeWhtPtImg=False, numWorkers=numWorkers)
        else:
            #use this to debug errors. Multiprocessing can sometimes hide the exact source of errors.
            completeBlkPtImg, completeWhtPtImg = cccObject.computePatchWiseBlackPointsFromTrack(tracksList=tracksList, srcVideosDir=inputVideoDir, gt_label=cowID, trackId=trackID, outImgDir=outputDir, patchSize=1, filterFrames=True, computeWhtPtImg=False)

        videoName = os.path.basename(inputVideoDir.rstrip('/'))
        if cowID is not None:
            illumMapSavePath = os.path.join(outputDir, f"{videoName}_illumMap_cowID_{cowID}.png")
        else:
            illumMapSavePath = os.path.join(outputDir, f"{videoName}_illumMap_trackID_{trackID}.png")
        
        cv2.imwrite(illumMapSavePath, completeBlkPtImg)
        print(f"Saved illumination map at: {illumMapSavePath}")

        #now extend the black point image (the illumination map) to cover parts of the unseen region
        extendedIllumMap = cccObject.extendBlackPointImg_normalizedGaussian(cBlkPtImg=completeBlkPtImg, outDir=outputDir, saveImg=False)

        cv2.imwrite(illumMapSavePath.replace('.png', '_extended.png'), extendedIllumMap)
        print(f"Saved extended illumination map at: {illumMapSavePath.replace('.png', '_extended.png')}")

    elif args.mode == 'extendOnly':
        inputIlluminationMapPath = args.inputIlluminationMapPath

        #load the illumination map
        cBlkPtImg = cv2.imread(inputIlluminationMapPath)

        outputDir = os.path.dirname(inputIlluminationMapPath)

        #extend the black point image (the illumination map) to cover parts of the unseen region
        extendedIllumMap = CowColorCorrector.extendBlackPointImg_normalizedGaussian(cBlkPtImg=cBlkPtImg, outDir=outputDir, saveImg=False)

        illumMapSavePath = inputIlluminationMapPath.replace('.png', '_extended.png')
        cv2.imwrite(illumMapSavePath, extendedIllumMap)
        print(f"Saved extended illumination map at: {illumMapSavePath}")




def parseArgs_forIlluminationMapComputation():
    parser = argparse.ArgumentParser(description='BMLP: Compute Illumination Maps from black cow tracks.')

    subparsers = parser.add_subparsers(dest='mode', help='Sub-commands for different modes of operation.')
    fullComputeParser = subparsers.add_parser('fullCompute', help='Compute the illumination map from tracks and video.')
    extendOnlyParser = subparsers.add_parser('extendOnly', help='Extend an existing illumination map without recomputing it.')

    # full compute args
    illuminationMapGrp = fullComputeParser.add_argument_group("Illumination Map Computation Parameters")
    illuminationMapGrp.add_argument('-t', '--tracksListPath', type=str, required=True, help='Path to the pickle file containing list of tracks, one of which is that of the required black cow.')
    illuminationMapGrp.add_argument('-i', '--inputVideoDir', type=str, required=True, help='Path to the directory containing input videos from which the tracks were computed.')
    illuminationMapGrp.add_argument('-o', '--outputDir', type=str, required=False, default='./', help='Path to the output directory where the illumination maps will be saved.')
    illuminationMapGrp.add_argument('-cfg', '--configPath', type=str, required=False, default='../../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml', help='Path to the config file for AutoCattlogger.')
    
    cowSelectGrp = illuminationMapGrp.add_mutually_exclusive_group(required=True)
    cowSelectGrp.add_argument('--cowID', type=str, default=None, help='ID of the black cow using which the illumination map has to be computed.')
    cowSelectGrp.add_argument('--trackID', type=int, default=None, help='Track ID of black cow uisng which the illumination map has to be computed.')

    otherGrp = fullComputeParser.add_argument_group("Other Parameters")
    otherGrp.add_argument('--useMultiProc', action='store_true', help='If set, use multiprocessing to speed up computation.')
    otherGrp.add_argument('--numWorkers', type=int, default=4, help='Number of workers to use if multiprocessing is enabled.')

    # extend only args
    extendOnlyParser.add_argument('-i', '--inputIlluminationMapPath', type=str, required=True, help='Path to the existing illumination map that has to be extended.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
        
    args = parseArgs_forIlluminationMapComputation()
    print(f"Args: {args}")
    computeIlluminationMaps(args)