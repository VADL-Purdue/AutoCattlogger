'''
Author: Manu Ramesh

This script has code to run inference using AutoCattleID (ACID).

'''
import cv2, pdb, glob

from matplotlib import pyplot as plt

from autoCattleID.ACID_inferenceEngine import build_ACID_InferenceEngine_from_config
from argparse import ArgumentParser


#############################################################################################################################
###################### FUNCTIONS TO RUN THE EVALUATIONS WITH DIFFERENT FILTER FUNCTIONS AND PARAMETERS ######################

def runInference(args):

    # Build inference engine and set its cattlog    
    inferenceEngine = build_ACID_InferenceEngine_from_config(configPath=args.configFilePath, pathOffsetVal='./') #pathOffsetVal should have path to top-level directory.
    inferenceEngine.setCattlog(tracksPath_train=args.tracksPath_Train, filterFnName=args.filterFnName, inclusionPercentage_top=0.2, cattlogDirPath=args.cattlogDirPath)

    if args.dataType == 'frame':
        # Run inference on single image/frame
        frame = cv2.imread(args.inputPath)
        if frame is None:
            raise ValueError(f"Could not read image from {args.inputPath}")
        
        inferenceEngine.inferFromFrame(frame=frame, saveDir=args.outDir, displayInferenceImg=args.displayImg)
    
    elif args.dataType == 'video':
        # Run inference on single video
        inferenceEngine.inferFromVideo_multiInstances(vidPath=args.inputPath, frameFreq=args.frameFreq, outRootDir=args.outDir, standalone=True)

    elif args.dataType == 'videoDir':
        # Run inference on all videos in a directory
        vidPathsList = glob.glob(f"{args.inputPath}/*.avi")
        vidPathsList.sort()

        if len(vidPathsList) == 0:
            raise ValueError(f"No .avi video files found in directory {args.inputPath}")

        if args.functionAsAutoCattlogger:
            inferenceEngine.functionAsAutoCattlogger() # sets flag to true

        inferenceEngine.inferFromVideosList_multiInstances(vidPathsList=vidPathsList, frameFreq=args.frameFreq, videosOutDir=args.outDir)
    
    else:
        raise ValueError(f"Invalid dataType {args.dataType}. Choose from 'frame', 'video', or 'videoDir'.")    


def parseArgs():
    parser = ArgumentParser(description="Evaluate AutoCattlog bit vectors (barcodes) for cow identification.")
    parser.add_argument('-C', '--configFilePath', type=str, required=True, help='Path to the AutoCattlogger config file for running inference using AutoCattleID.')

    srcGrpParser = parser.add_mutually_exclusive_group(required=True)
    srcGrpParser.add_argument('-t', '--tracksPath_Train', type=str, default=None, help='Path to the training tracks pickle file.')
    srcGrpParser.add_argument('-c', '--cattlogPath', type=str, default=None, help='Path to the Cattlog (cow data matrix) dict file.')

    parser.add_argument('--cattlogDirPath', type=str, required=True, default=None, help='Path to directory that has iconic/canonical cow images for each cowID in the cattlog.')

    parser.add_argument('--filterFnName', type=str, required=True, default=None, help='Name of the track point filtering function to use during inference. Choose from available options. Cannot be applied if you supply the cattlog directly. Eg: trackPtsFilter_byProximityOfRBboxToFrameCenter')
    parser.add_argument('-o', '--outDir', type=str, default='../outputs/ACID_outputs/', help='Directory to save the outputs.')

        

    subparsers = parser.add_subparsers(dest='dataType', help='Type of data to run inference on. Options are: frame, video, videoDir.' )

    # Sub-parser for image/frame data
    parser_frame = subparsers.add_parser('frame', help='Run inference on a single image/frame.')
    parser_frame.add_argument('-i', '--inputPath', type=str, required=True, help='Path to the input image/frame file.')
    parser_frame.add_argument('-d', '--displayImg', action='store_true', help='If set, display the inference image.')

    # Sub-parser for video data
    parser_video = subparsers.add_parser('video', help='Run inference on a single video file.')
    parser_video.add_argument('-i', '--inputPath', type=str, required=True, help='Path to the input video file.')
    parser_video.add_argument('-F', '--frameFreq', type=int, required=False, help='Frame frequency for processing videos. One in every frameFreq frames will be processed.', default=1)

    # Sub-parser for video directory data
    parser_videoDir = subparsers.add_parser('videoDir', help='Run inference on all video files in a directory.')
    parser_videoDir.add_argument('-i', '--inputPath', type=str, required=True, help='Directory containing (only) the top-view videos to process.')
    parser_videoDir.add_argument('-F', '--frameFreq', type=int, required=False, help='Frame frequency for processing videos. One in every frameFreq frames will be processed.', default=1)
    parser_videoDir.add_argument('--functionAsAutoCattlogger', action='store_true', help='If set, the inference engine will function as an AutoCattlogger and generate cattlog outputs similar to the AutoCattlogger. Useful if you want to clean up annotations.')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    args = parseArgs()
    runInference(args)

    

