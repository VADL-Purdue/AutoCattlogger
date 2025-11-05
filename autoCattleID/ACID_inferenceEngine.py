'''
Author: Manu Ramesh

This module has code to run AutoCattleID (ACID) inference on data.

'''
import cv2, pickle, pdb, glob, os, numpy as np, sys, pdb

sys.path.insert(0, '../')
from autoCattlogger.helpers.helper_for_QR_and_inference import inferBitVector, findSameBitVecCows
from autoCattlogger.helpers.helper_for_infer import printAndLog
from autoCattlogger.autoCattlogger import AutoCattloger

from autoCattleID.helpers.helper_for_trackPointFiltering import getFilteredCattlog, getFilterFnFromName
from autoCattlogger.utils.display_utils import get_cow_matches_frame


from collections import Counter
from tqdm import tqdm, trange

import logging, yaml
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count
import multiprocessing
import pandas as pd
from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors

import matplotlib.pyplot as plt

from detectron2.config import get_cfg
import torch

from autoCattlogger.helpers.helper_for_tracking import getNewTrackPoint, matchTrackPoints

from autoCattlogger.helpers.helper_for_infer import log_cow_det_reid_stat

#for reading frames from hdf5 files
from autoCattlogger.helpers.helper_for_hdf5 import VideoCapture_HDF5

#for postprocessing tracks
from autoCattlogger.helpers.helper_for_autoCattlog import postProcessTracks1, postProcessTracks_RL


class ACID_InferenceEngine(AutoCattloger):

    def __init__(self, *args, **kwargs):

        '''
        Init fn for ACID Inference engine. Child of AutoCattlogger class.
        Takes in arguments taken in by the AutoCattlogger class.
        
        :param self: Description
        :param args: Description
        :param kwargs: Description
        '''

        super().__init__(*args, **kwargs)
        self.cattlog = None 
        self.cattlogDirPath = None #path to directory that has iconic/canonical cow images for each cowID in the cattlog, used to create cow matches display frames during inference

        # Realized that I built a better AutoCattlogger when building this inference engine.
        # You can use it to build cattlogs from a new day, while also saving predictions on the fly.
        # This will be of great help when you want to weed out errors in the ground truth labels of training data.
        # So, instead of deleting the Cattlogging functionality from this class, I am making an option that can be turned on if needed.
        # You can set it using the function functionAsAutoCattlogger() below. This is like a cheat code to turn on AutoCattlogger functionality on/off in this class.
        self.saveAutoCattloggerOutputs = False #whether to save the autoCattlogger outputs like bit vectors, cropped images etc. during inference from video


    def functionAsAutoCattlogger(self, value=True):
        '''
        Sets whether to save AutoCattlogger outputs during inference from video.
        Turn this on and run ACID Inference to save predicted IDs along with the tracks information. Helps you annotate the ground truth information easily.
        :param value: True/False
        '''

        self.saveAutoCattloggerOutputs = value

    def setCattlog(self, cattlogPath=None, tracksPath_train=None, filterFnName=None, cattlogDirPath=None, **kwargs):
        '''
        Load the cattlog from the given path.
        Otherwise, generates the cattlog from the given training tracks path.

        If you choose to generate the cattlog from the training tracks, you can also provide a filter function name to filter the trackpoints used to generate the cattlog. This will generate better quality barcodes in the cattlog.

        NOTE: The tracks at the supplied tracksPath_train must have ground truth cowIDs attached. The code looks for the 'gt_label' key in each track dictionary to get the ground truth cowID.

        Inputs:
            cattlogPath: path to the cattlog pickle file.
            tracksPath_train: path to the training tracks pickle file.
            filterFnName: name of the filter function to filter trackpoints used to generate the cattlog. For options, check the getFilterFnFromName() fn in helpers/helper_for_trackPointFiltering.py module. Paper uses trackPtsFilter_byProximityOfRBboxToFrameCenter with inclusion percentage of 20%.
            cattlogDIrPath: path to directory that has iconic/canonical cow images for each cowID in the cattlog, used to create cow matches display frames during inference.
            kwargs: Other keyword arguments such as inclusionPercentage_top (actually a proportion) that are passed to the trackpoint filtering function.

        Outputs:
            None
            Sets the cattlog for the object once.
        '''

        assert cattlogPath is not None or tracksPath_train is not None, "Either cattlogPath or tracksPath_train must be provided to generate the training cattlog."

        if cattlogPath is not None:
            self.cattlog = pickle.load(open(cattlogPath, "rb"))
            printAndLog(f"Loaded cattlog from {cattlogPath} with {len(self.cattlog)} entries.\n", printDebugInfoToScreen=True)
        else:
            tracksList = pickle.load(open(tracksPath_train, "rb"))
            printAndLog(f"Loaded {len(tracksList)} training tracks from {tracksPath_train}.\n", printDebugInfoToScreen=True)
            
            # Filter the trackpoints and create a cattlog if requested
            trackPtsFilterFn = getFilterFnFromName(filterFnName) # will be None if filterFnName is None
            self.cattlog = getFilteredCattlog(tracksList, trackPtsFilterFn=trackPtsFilterFn, **kwargs) #will return the unfiltered cattlogDict if trackPtsFilterFn is None

        assert cattlogDirPath is not None, "cattlogDirPath must be provided to setCattlog(). This is required to populate the inference frame with a sample image of cow whose ID is the predicted ID."
        self.cattlogDirPath = cattlogDirPath

    
    def inferFromFrame(self, frame, sortTopK_byWarpingCost=False, frameCount=0, saveDir=None, displayInferenceImg=True):
        '''
        Run inference on a single frame (image) and return the predicted cowIDs.
        Displays the inference frame if needed, saves the image if needed.


        Inputs:
            frame: The input image frame on which inference is to be run.
            sortTopK_byWarpingCost: Boolean flag to sort top K predictions by warping cost. Unused option.
            frameCount: The frame count to be displayed in the inference frame.
            saveDir: Directory in which the inference frame should be saved. If None, the inference image is not saved.
            displayInferenceImg: Boolean flag to display the inference image.

        Outputs: 
            predCowIDs: List of tuples containing maskNumber and predicted cowID for each detected cow in the frame.
        '''

        assert self.cattlog is not None, "Cattlog is not set. Please set the cattlog using setCattlog() before running inference."

        # set up logging params
        _countsDict = {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox': 0, 'total_instanceBBoxes': 0,
                        'total_AR_area_pass': 0, 'total_allKps_det_inter': 0, 'total_allKPRulePass': 0,
                        'total_correct_preds': 0, 'total_kpCorrected_instances': 0,
                        'gt_locs_list': []} 
        kpRulesPassCounts = np.array([0] * 100, int)  # to hold a histogram of number of KP rules passed
        nKPDetectionCounts = np.array([0] * (len(self.keypoint_names) + 1), int)  # histogram of number of keypoints detected per image, +1 to include 0 - no visible kp case

        # Verified. The self.getBitVectorsFromFrame() fn takes checks the self.colorCorrectionOn flag and does color correction if needed.
        # This flag can be set/unset in the AutoCattlogger config file.
        
        bitVecStrList, kpCorrectionMethodList, cowAutoCatOutImg, cropList, rotatedBBOX_img = self.getBitVectorsFromFrame(frame, _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, nKPDetectionCounts=nKPDetectionCounts, frameCount=0, gt_label=None, doNotComputeCowAutoCatOutImg=True)

        # infer cow IDs from bit vectors for each cow detected in the frame, then overlay and display them
        predCowIDs = []
        for entryDict in bitVecStrList:
            maskNumber = entryDict['maskNumber'] #used to index the cows in the frame
            bitVecStr = entryDict['bitVecStr']['blk16'] #use the value with block size 16 # This is the only size supported for now.
            predicted_cowID, pred_confidence, _ = inferBitVector(self.cattlog, queryBitVector=bitVecStr, gt_cowID=None, sortTopK_byWarpingCost=sortTopK_byWarpingCost)
            
            predCowIDs.append((maskNumber, predicted_cowID))
            
    
        printAndLog(f"Predicted cowIDs: {predCowIDs}\n", printDebugInfoToScreen=True)

        # Display the Predicted cowIDs on the cows at the center of the rotated bounding boxes
        for prediction in predCowIDs:
            maskNumber = prediction[0]
            predicted_cowID = prediction[1]
            # find the center of the rotated bbox for this maskNumber
            # the rotated bbox pts are in self.rotatedBboxPtsList in the same order as the maskNumbers

            for bbox in self.rotatedBboxPtsList:
                # put the cowID value in the center
                text = f'PredID: {predicted_cowID}'
                cx, cy = (bbox[:, 0].mean(), bbox[:, 1].mean())
                bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)

                # rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30), (int(cx) + 500, int(cy) + 15), (0, 0, 0), -1)  # background rectangle
                rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30), (int(cx) + 315, int(cy) + 15), (0, 0, 0), -1)  # background rectangle
                
                font = cv2.FONT_HERSHEY_SIMPLEX;
                fontScale = 1;
                fontColor = (255, 255, 255);
                thickness = 2;
                lineType = 2
                cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                            lineType)
                
            
        # Create cow matches display frame - to show the frame number, predicted cow and the image of the cow whose ID = predicted cowID

        cowRecogEvalImg = get_cow_matches_frame(inFrame=rotatedBBOX_img, predIdList=[x[1] for x in predCowIDs], cropList=cropList,
                                            cattlogDirPath=self.cattlogDirPath, maxCows=3,
                                            frameNumber=frameCount)  # now takes images from top view KP aligned cattlog


        # Reset Values (Defined in the AutoCattlogger class)
        self.predictions = None
        self.rotatedBboxPtsList = []
        
        if saveDir is not None:
            os.makedirs(saveDir, exist_ok=True)
            savePath = os.path.join(saveDir, f"inference_frame_{frameCount:05d}.jpg")
            cv2.imwrite(savePath, cv2.cvtColor(cowRecogEvalImg, cv2.COLOR_RGB2BGR))
            printAndLog(f"Saved inference display image to {savePath}\n", printDebugInfoToScreen=True)

        if displayInferenceImg:
            plt.imshow(cowRecogEvalImg)
            plt.axis('off')
            plt.show()            

        return predCowIDs
    
    def inferFromVideo_multiInstances(self, vidPath, cv2VideoWriterObj=None, frameFreq = 3, outRootDir="./ACID_MultiCow_outputs/", frame_width = None, frame_height = None, _countsDict=None, kpRulesPassCounts=None, kpRules_maxPossibleConf=None, _nKPDetectionCounts=None, openTracks=[], closedTracks=[], startingTrackID=0, noDetFramesLimit=6, standalone=False, postProcessingFn=None, sortTopK_byWarpingCost=False):
        '''
        Track based inference.
                    
        Computes cattlog bitVector from any video - even uncut video.
        Can track multiple cows in the same frame.

        Inputs: 
            vidPath: Path to the video file on which inference must be run.
            frameFreq: picks one in every frameFreq frame for processing - proceeds to processing only if frameCount % frameFreq = 0 (this is not FPS)
            outRootDir: Root directory where output files will be saved.
            frame_width: Width of the video frames.
            frame_height: Height of the video frames.
            openTracks: tracks that are open and will be updated. You can pass in the list of open tracks here.
            closedTracks: tracks that are closed and will not be updated anymore. This might not be necessary but it just makes coding easier. You can change this later.
            startingTrackID: The starting ID to assign to the next new track.
            noDetFramesLimit: the number of contigious frames without any detections to wait before ending all the open tracks. We account for frameFreq below.
            standalone: If true, prints and logs stats of video file.
            postProcessingFn: optional post processing function to apply to the tracks after they are generated.

            sortTopK_byWarpingCost: If true, sorts the top K matches by warping cost. Currently unused.

            # Inputs that were also in the cattlogging fns
            kpRules_maxPossibleConf: Maximum possible confidence for keypoint rules. (= total number of keypoint rules as each rule pass provides 1 confidence point)
            _nKPDetectionCounts: A list to store the histogram of the number of keypoints detected per instance. The user need not worry about it.
            _countsDict: A dictionary with count values for storing metrics of intermediate results.

        Outputs:
            openTracks: The list of open tracks after processing the video. Has any cow that has still not left the scene.
            closedTracks: The list of closed tracks after processing the video. Has tracks of all cows (and what the detector thought were cows, i.e. false positives).
            trackId: An integer value = total of all cow tracks found +1. This is the track id that needs to be assigned to the next track that is found by the detector/tracker (to be passed as the startingTrackID to the next call of this function on the next video in sequence).

        '''

        assert self.cattlog is not None, "Cattlog is not set. Please set the cattlog using setCattlog() before running inference."

        if standalone:

            os.makedirs(outRootDir, exist_ok=True)

            # need to create a new log file if these functions are called again from the same object and the caller requests a different filename
            # Remove all handlers associated with the root logger object.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=f'{outRootDir}/log_ACID_inference_OnVideoMultiCow_QR.log',
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
            out = cv2.VideoWriter(f"{outRootDir}/ACID_inference_out_standalone_vid_{videoName.split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        else:
            #out = cv2.VideoWriter(f'{outRootDir}/autoCattlog_out_vid_{gt_label}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
            #out = cv2.VideoWriter(f'{outRootDir}/autoCattlog_out_vid_{gt_label}.mkv', -1, 30, (frame_width, frame_height))
            out = cv2.VideoWriter(f"{outRootDir}/ACID_inference_out_vid_{videoName.split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height)) #codec_id 001B

        
        if standalone: print(f"\n\nProcessing video {vidPath.split('/')[-1]}\n\n")

        

        frameCount = 0

        # inference stats setup for standalone mode
        if _countsDict is None:
            _countsDict = {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox': 0, 'total_instanceBBoxes': 0,
                        'total_AR_area_pass': 0, 'total_allKps_det_inter': 0, 'total_allKPRulePass': 0,
                        'total_correct_preds': 0, 'total_kpCorrected_instances': 0,
                        'gt_locs_list': []}  # change later ot total_correctPredictions if possible
        if kpRulesPassCounts is None:
            kpRulesPassCounts = np.array([0] * 100, int)  # to hold a histogram of number of KP rules passed
        # if kpRules_maxPossibleConf is None: #redundant
        #    kpRules_maxPossibleConf = None
        if _nKPDetectionCounts is None:
            _nKPDetectionCounts = np.array([0] * (len(self.keypoint_names) + 1), int)  # histogram of number of keypoints detected per image, +1 to include 0 - no visible kp case

        cow_bitVecStrList = [] #list of all bit vectors for a given cow from all frames of the video


        #TRACKS:
        trackId = startingTrackID #trackId for the next track to be created
        
        #to enable quick ending of tracks when there are no detections
        noDetFramesLimit = noDetFramesLimit//frameFreq #to account for frames skipped
        noDetFramesCounter = 0
        

        # For Reference:
        # track = {'trackId': trackId, 'startFrame':-1, 'endFrame':-1, 'hasCow':False, 'sampleCropImg':None, 'trackPoints':[], 'trackLvlPred': 'str', 'pred_countsDict': pred_countsDict}
        # trackPoint = {'videoName': videoName, 'frameNumber':frameNumber, 'rotatedBBOX_locs':rotatedBBOX_locs, 'bitVecStr':'', 'kpCorrectionMethodUsed':'', 'instLvlPred': 'str'}
        # matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount, iouThresh=0.3, forInference=False by default):
    
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
            _countsDict['total_seenFrames'] += 1

            # if not self.useMMPOSE:
            #    img_prediction_op = self.predictOnImage_QR(frame=frame, _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, _nKPDetectionCounts=_nKPDetectionCounts, frameCount=frameCount, gt_label=gt_label)
            # else:

            cattlog_output = self.getBitVectorsFromFrame(frame=frame, _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, _nKPDetectionCounts=_nKPDetectionCounts, frameCount=frameCount, gt_label=None, doNotComputeCowAutoCatOutImg=True) #force gt_label to be None
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
                    currentTrackPoints.append(getNewTrackPoint(videoName=vidPath.split('/')[-1], frameNumber=frameCount, rotatedBBOX_locs=rotatedBboxPts, forInference=True))


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
                openTracks, closedTracks, trackId = matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount=frameCount, forInference=True)

                #Do some operations on the open tracks.
                for track in openTracks:

                    #this is where you compute track level inference
                    
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

                    # Infer instance level cowID from the bit-vector for the latest trackPt, 
                    # use it to update the track level prediction counts, and compute the track level prediction
                    latestBitVecStr = track['trackPoints'][-1]['bitVecStr']
                    if latestBitVecStr != '':
                        predicted_cowID, pred_confidence, _ = inferBitVector(self.cattlog, queryBitVector=latestBitVecStr, gt_cowID=None, sortTopK_byWarpingCost=sortTopK_byWarpingCost)
                        
                        track['trackPoints'][-1]['instLvlPred'] = predicted_cowID

                        #update the pred_countsDict
                        track['pred_countsDict'][predicted_cowID] = track['pred_countsDict'].get(predicted_cowID, 0) + 1

                        #update the track level prediction - cowID with the maximum counts
                        track['trackLvlPred'] = max(track['pred_countsDict'], key=track['pred_countsDict'].get) # directly using max without key should also work

                        # log these predictions
                        printAndLog(f"Predictions: Instance Level: {predicted_cowID}, Track Level: {track['trackLvlPred']}\n", printDebugInfoToScreen=self.printDebugInfoToScreen)
                    
                    #display the instance and track level predictions on the cow
                    if track['trackLvlPred'] != '????': # To avoid displaying predictions for stray detections

                        text = f"(Inst:{track['trackPoints'][-1]['instLvlPred']} | Track:{track['trackLvlPred']})"
                        bbox = trackPt['rotatedBBOX_locs']
                        cx, cy = (bbox[:, 0].mean(), bbox[:, 1].mean()) #center of the rotated bbox
                        cy = cy + 45 # offset cy by a few pixels to avoid overlapping with previous text
                        bottomLeftCornerOfText = (int(cx), int(cy))  # (10, 500)
                        rotatedBBOX_img = cv2.rectangle(rotatedBBOX_img, (int(cx) - 5, int(cy) - 30), (int(cx) + 315, int(cy) + 15), (0, 0, 0), -1)  # background rectangle
                        # cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 2)
                        cv2.putText(rotatedBBOX_img, text, bottomLeftCornerOfText, cv2.FONT_HERSHEY_SIMPLEX, 0.8 ,(0, 191, 255), 2, 2)


                self.rotatedBboxPtsList = [] #[[0,0], [0,0], [0,0], [0,0]] #None #resetting the bbox points for the next frame
                self.predictions = None #resetting the predictions for the next frame


                # cowAutoCatOutImg = get_cow_autocattlog_frame(inFrame=rotatedBBOX_img, gtLabelsList=[gt_label]*len(cropList), cropList=cropList, maxCows=5, frameNumber=frameCount)  # now takes images from top view KP aligned cattlog

                # predIdList = [track['trackLvlPred'] for track in openTracks if track['hasCow']]  # for showing track level preds
                predIdList = [track['trackPoints'][-1]['instLvlPred'] for track in openTracks if track['hasCow']]  # for showing instance level preds
                predIdList = [x for x in predIdList if x != '????']  # removing empty preds
                cowRecogEvalImg = get_cow_matches_frame(inFrame=rotatedBBOX_img, predIdList=predIdList, cropList=cropList,
                                            cattlogDirPath=self.cattlogDirPath, maxCows=3,
                                            frameNumber=frameCount) 

                # it will be none if no instances are found in it (i am doing this to reduce the output file size - change it to send a dummy frame without predictions if you want to keep the frame)
                out.write(cv2.resize(cowRecogEvalImg, (frame_width, frame_height)))

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

                        openTracks, closedTracks, trackId = matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount=frameCount, forInference=True)
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
            log_cow_det_reid_stat(_countsDict.copy(), kpRulesPassCounts, _nKPDetectionCounts)
            
            
            #close all open tracks
            logging.info(f"Closing all open tracks. There are {len(openTracks)} open tracks.")

            for track in openTracks:
                track['endFrame'] = track['trackPoints'][-1]['frameNumber']
                closedTracks.append(track)

            openTracks = [] #reset the open tracks


            #SAVE SAMPLE CROP IMAGES
            #Saving all crops - to just see which all cows have appeared
            os.makedirs(f"{outRootDir}/ACID_sampleCrops_fromAllClosedTracks/", exist_ok=True)
            for track in closedTracks:
                if track['hasCow']:
                    cv2.imwrite(f"{outRootDir}/ACID_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}_predID_{track['trackLvlPred']}.jpg", track['sampleCropImg'])

            if self.saveAutoCattloggerOutputs:
                os.makedirs(f"{outRootDir}/autoCattlog_sampleCrops_fromAllClosedTracks/", exist_ok=True)
                for track in closedTracks:
                    if track['hasCow']:
                        cv2.imwrite(f"{outRootDir}/autoCattlog_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])

            # POST PROCESS AND SAVE REQUIRED TRACKS AND BIT VECTORS
            if postProcessingFn is not None:
                #PROCESSING THE BIT VECTORS
                requiredTracks, bitVecCattlogDict = postProcessingFn(closedTracks, frameW=frame_width, printDebugInfoToScreen=True)     


                #save sample crop images only for required cows
                if self.saveAutoCattloggerOutputs:
                    #save sample crop images only for required cows
                    os.makedirs(f"{outRootDir}/autoCattlog_sampleCrops_fromRequiredTracks/", exist_ok=True)
                    for track in requiredTracks:
                        cv2.imwrite(f"{outRootDir}/autoCattlog_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])

                # SAVE ACID Outputs                    
                os.makedirs(f"{outRootDir}/ACID_sampleCrops_fromRequiredTracks/", exist_ok=True)
                for track in requiredTracks:
                    cv2.imwrite(f"{outRootDir}/ACID_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}_predID_{track['trackLvlPred']}.jpg", track['sampleCropImg'])
                    del track['sampleCropImg'] #deleting to save space  
                
                #SAVE THE NECESSARY FILES
                if self.saveAutoCattloggerOutputs:
                    pickle.dump(requiredTracks, open(f"{outRootDir}/requiredTracks.pkl", 'wb'))
                    pickle.dump(bitVecCattlogDict, open(f"{outRootDir}/cowDataMatDict_autoCattlogMultiCow_blk16{'_CC' if self.colorCorrectionOn else ''}.p", 'wb'))


            # SAVE EVERYTHING ELSE

            #delete the sampleCropImg key from the dictionary to save space
            for track in closedTracks:
                if 'sampleCropImg' in track: #the requiredTracks would have saved tracks by reference. So, some tracks might already have the sampleCropImg deleted from above.
                    del track['sampleCropImg']

            #SAVE THE NECESSARY FILES
            if self.saveAutoCattloggerOutputs:
                pickle.dump(openTracks, open(f"{outRootDir}/openTracks.pkl", 'wb')) #this should be empty
                pickle.dump(closedTracks, open(f"{outRootDir}/closedTracks.pkl", 'wb'))

        return openTracks, closedTracks, trackId

    def inferFromVideosList_multiInstances(self, vidPathsList, frameFreq=3, videosOutDir = "./ACID_fromVideoList_multiInstances_outputs/", frame_width = None, frame_height = None, logDir=None, logSuffix='', postProcessingFn=postProcessTracks1):
        '''
        Takes in list of unsegmented videos (hour-long videos) and proceses them to create a cattlog for each cow in each video.

        I call a separate post-processing function to create the final cattlog bit vectors. 
        In that function, I can filter out cows that walk in the opposite direction. 
        I can combine the GT labels with the track IDs in another function.

        
        Inputs:
            vidPathsList: list of paths to video files (*.avi)
            frameFreq: picks one in every frameFreq frame for processing - proceeds to processing only if frameCount % frameFreq = 0 (this is not FPS)
            videosOutDir: directory to save the output video files
            frame_width: width of the video frames
            frame_height: height of the video frames
            logDir: directory to save the log files
            logSuffix: suffix to add to the log file name
            postProcessingFn: optional post processing function to apply to the tracks after they are generated. I have defined a few to filter out tracks that do not have any cows in them (using the hasCow flag), and those that walk in unwanted directions.

        Outputs:
            :return: None
        
        :saved files in logDir:
            the log file
            openTracks: list of open tracks
            closedTracks: list of closed tracks
            requiredTracks: list of required tracks (the tracks after post-processing i.e. filtering out unnecessary tracks of cows moving in the wrong directions)
            bitVecCattlogDict: dictionary of bit vectors (serialized cow barcodes) for each cow in each video. The keys are the GT labels.
            Sample cropped, template aligned images: in {logDir}/autoCattlog_sampleCrops (Only for cows in required tracks. - Change it to save samples for all cows found if it can help you in cleaning data from a new dataset - to make one-to-one matches with the video output and the CSV file in the dairy or whatever.)
        '''

        assert self.cattlog is not None, "Cattlog is not set. Please set the cattlog using setCattlog() before running inference."

        bitVecCattlogDict = {} #we are only processing blkSize 16 as of now

        #create videosOutDir if it doesn't exist
        os.makedirs(videosOutDir, exist_ok=True)

        if logDir is None:
            logDir = videosOutDir #just to keep them all in one place

        # need to create a new log file if these functions are called again from the same object and the caller requests a different filename
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f'{logDir}/log_ACID_inference_OnVideoSet_multiInstances_QR{logSuffix}.log', encoding='utf-8', level=logging.DEBUG, filemode='w', format='%(asctime)s>> %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')  # set up logger

        logging.info(f"Creating cattlog (multi-instances) on video-list.")
        self.logInputParams() #Saving all input params in the log file for - logging purposes ¯¯\__(O_O)__/¯¯

        logging.info(f"Processing video list")

        # saving video
        #out = cv2.VideoWriter('autocattlog_vidList_multiInstnaces_out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        # inference stats setup
        _countsDict = {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox': 0, 'total_instanceBBoxes': 0, 'total_AR_area_pass': 0, 'total_allKps_det_inter': 0, 'total_allKPRulePass': 0, 'total_correct_preds': 0, 'total_kpCorrected_instances':0, 'gt_locs_list': [], 'total_correct_pred_vids':0, 'topK_VideoLevelDict':{}, 'nVideos':len(vidPathsList), 'cowsWithNoCorrectPreds_list':[]}
        kpRulesPassCounts = np.array([0] * 100, int)  # to hold a histogram of number of KP rules passed
        kpRules_maxPossibleConf = None
        _nKPDetectionCounts = np.array([0] * (len(self.keypoint_names) + 1), int)  # histogram of number of keypoints detected per image, +1 to include 0 - no visible kp case

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

            #_countsDict and other such variables are passed by reference
            #cowBitVecStr = self.cattlogFromCutVideo(vidPath=vidPath, cv2VideoWriterObj=out, frameFreq=frameFreq, outRootDir=videosOutDir, frame_width=1920, frame_height=1080, gt_label=gt_label, _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, _nKPDetectionCounts=_nKPDetectionCounts, standalone=False)
            #bitVecCattlogDict[gt_label] = {'blk16':cowBitVecStr}

            openTracks, closedTracks, trackID = self.inferFromVideo_multiInstances(vidPath, cv2VideoWriterObj=None, frameFreq = frameFreq, outRootDir=videosOutDir, frame_width = frame_width, frame_height = frame_height, gt_label=None, _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, _nKPDetectionCounts=_nKPDetectionCounts, openTracks=openTracks, closedTracks=closedTracks, startingTrackID=trackID, noDetFramesLimit=6, standalone=False)


        # Using Joblib for parallel evaluation
        # Do not use this. Parallel evaluation messes up the logs. There will be nothing useful in the logfile. However, the evaluated videos turned out to be fine - so you could use this to get all evaluated cut-videos with overlays if you are in a hurry as this definitely runs faster than serial evaluation with the for loop.
        #_ = Parallel(n_jobs=8)(delayed(self.predictOnVideo2_QR)(vidPath=vidPath, cv2VideoWriterObj=None, frameFreq=frameFreq, outRootDir=videosOutDir, frame_width=1920, frame_height=1080, gt_label=gt_labels_list[idx], _countsDict=_countsDict, kpRulesPassCounts=kpRulesPassCounts, kpRules_maxPossibleConf=kpRules_maxPossibleConf, _nKPDetectionCounts=_nKPDetectionCounts, standalone=False) for idx, vidPath in enumerate(pbar)) # messes up the logs

        #out.release()

        
        #*************************************************************************************************************************#
        #LEGACY OVERHEADS
        if True: #just to allow me to collapse the code block
            total_correct_pred_vids = _countsDict.pop('total_correct_pred_vids') #must remove this value before passing to log_cow_det_reid_stat function
            if self.kpRules_maxPossibleConf is not None: kpRulesPassCounts = kpRulesPassCounts[:self.kpRules_maxPossibleConf + 1]  # trimming the histogram's X axis, +1 as count starts from 0 where 0 indicates that all 10 kps are not detected
            log_cow_det_reid_stat(_countsDict.copy(), kpRulesPassCounts, _nKPDetectionCounts)
            
            if self.saveAutoCattloggerOutputs:
                pickle.dump({'_countsDict': _countsDict.copy(), 'kpRulesPassCounts': kpRulesPassCounts, '_nKPDetectionCounts' : _nKPDetectionCounts}, open(f'{logDir}/metricsDict_autocattlog_onVideoSet_QR{logSuffix}.p','wb'))

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
        os.makedirs(f"{videosOutDir}/ACID_sampleCrops_fromAllClosedTracks/", exist_ok=True)
        for track in closedTracks:
            if track['hasCow']: #otherwise the image will be None
                cv2.imwrite(f"{videosOutDir}/ACID_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}_pred_{track['trackLvlPred']}.jpg", track['sampleCropImg'])
        
        if self.saveAutoCattloggerOutputs:
            os.makedirs(f"{videosOutDir}/autoCattlog_sampleCrops_fromAllClosedTracks/", exist_ok=True)
            for track in closedTracks:
                if track['hasCow']: #otherwise the image will be None
                    cv2.imwrite(f"{videosOutDir}/autoCattlog_sampleCrops_fromAllClosedTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])
        

        # POST PROCESS AND SAVE BIT VECTORS IF A POST PROCESSING FN IS PROVIDED
        if postProcessingFn is not None:

            #PROCESSING THE BIT VECTORS
            requiredTracks, bitVecCattlogDict = postProcessingFn(closedTracks, frameW=frame_width, printDebugInfoToScreen=True)    

            #save sample crop images only for required cows
            if self.saveAutoCattloggerOutputs:
                os.makedirs(f"{videosOutDir}/autoCattlog_sampleCrops_fromRequiredTracks/", exist_ok=True)
                for track in requiredTracks:
                    cv2.imwrite(f"{videosOutDir}/autoCattlog_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}.jpg", track['sampleCropImg'])

            # SAVE ACID Outputs
            os.makedirs(f"{videosOutDir}/ACID_sampleCrops_fromRequiredTracks/", exist_ok=True)
            for track in requiredTracks:
                cv2.imwrite(f"{videosOutDir}/ACID_sampleCrops_fromRequiredTracks/trackId_{track['trackId']}_pred_{track['trackLvlPred']}.jpg", track['sampleCropImg'])
                del track['sampleCropImg'] #deleting to save space  


            if self.saveAutoCattloggerOutputs:
                pickle.dump(requiredTracks, open(f"{logDir}/requiredTracks.pkl", 'wb'))
                pickle.dump(bitVecCattlogDict, open(f"{logDir}/cowDataMatDict_autoCattlogMultiCow_blk16{'_CC' if self.colorCorrectionOn else ''}.p", 'wb'))

        ############################################################
        #SAVE EVERYTHING ELSE NOW
        
        if self.saveAutoCattloggerOutputs:
            #delete the sampleCropImg key from the dictionary to save space
            for track in closedTracks:
                if 'sampleCropImg' in track: #the requiredTracks would have saved tracks by reference. So, some tracks might already have the sampleCropImg deleted from above.
                    del track['sampleCropImg']

            #SAVE THE NECESSARY FILES
            pickle.dump(openTracks, open(f"{logDir}/openTracks.pkl", 'wb')) #should be useless now
            pickle.dump(closedTracks, open(f"{logDir}/closedTracks.pkl", 'wb'))
        

def build_ACID_InferenceEngine_from_config(ac_config=None, configPath=None, pathOffsetVal="../"):
    '''
    Utility function to build AutoCattlogger object from a config file.
    :param configPath: path to the config file (YAML format)
    :return: AutoCattlogger object
    '''

    assert (ac_config is not None) or (configPath is not None), "Either ac_config or configPath must be provided."

    if ac_config is None:
        ac_config = yaml.safe_load(open(configPath, 'r'))

    detectron2_device = int(ac_config['detectron2']['device']) if torch.cuda.is_available() else 'cpu' #= 0 if only 1 gpu is available

    # Must offset paths by one directory level up to the top level directory since this file is in autoCattleID folder.
    # All paths in the config file are relative to the top level directory.

    # get detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(pathOffsetVal + ac_config['detectron2']['model_config_path'])
    cfg.MODEL.DEVICE = detectron2_device
    cfg.MODEL.WEIGHTS = pathOffsetVal + ac_config['detectron2']['weights_path']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ac_config['detectron2']['MODEL_ROI_HEADS_SCORE_THRESH_TEST']

    #mmpose config
    useMMPOSE = ac_config['mmpose']['useMMPOSE'] #should be True
    mmpose_poseConfigPath = ac_config['mmpose']['mmpose_poseConfigPath']
    mmpose_modelWeightsPath = ac_config['mmpose']['mmpose_modelWeightsPath']
    mmpose_device = ac_config['mmpose']['device'] if torch.cuda.is_available() else 'cpu'

    # pdb.set_trace(header="Building AutoCattlogger object from config")

    # Build ACID_InferenceEngine object from AutoCattlogger config.
   
    inferenceEngine = ACID_InferenceEngine(cfg=cfg,
                              affectedKPsDictPath=pathOffsetVal + ac_config['AutoCattlogger']['affectedKPsDictPath'],
                              datasetKpStatsDictPath=pathOffsetVal + ac_config['AutoCattlogger']['datasetKpStatsDictPath'],
                              cowAspRatio_ulim=ac_config['AutoCattlogger'].get('cowAspRatio_ulim', 0.53),
                              cowAreaRatio_llim=ac_config['AutoCattlogger'].get('cowAreaRatio_llim', 0.042),
                              saveHardExamples=ac_config['AutoCattlogger'].get('saveHardExamples', False),
                              saveEasyExamples=ac_config['AutoCattlogger'].get('saveEasyExamples', False),
                              printDebugInfoToScreen=ac_config['AutoCattlogger'].get('printDebugInfoToScreen', False),
                              keypointCorrectionOn=ac_config['AutoCattlogger'].get('keypointCorrectionOn', True),
                              colorCorrectionOn=ac_config['AutoCattlogger'].get('colorCorrectionOn', True),
                              blackPointImgPth=ac_config['AutoCattlogger'].get('blackPointImgPth', None),
                              useMMPOSE=useMMPOSE,
                              mmpose_poseConfigPath=pathOffsetVal + mmpose_poseConfigPath,
                              externalCC_fn=None, #ac_config['AutoCattlogger']['externalCC_fn'], #force None for now
                              threshVal_CC=None if ac_config['AutoCattlogger'].get('threshVal_CC', "None") == "None" else int(ac_config['AutoCattlogger']['threshVal_CC']),
                              mmpose_modelWeightsPath=pathOffsetVal + mmpose_modelWeightsPath,
                              mmpose_device=mmpose_device)


    return inferenceEngine


if __name__ == '__main__':

    inferenceEngine = build_ACID_InferenceEngine_from_config(configPath="../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml")

    tracksPath_train="../outputs/temp/tracks_withGTLabels.pkl"
    cattlogDirPath="../outputs/temp/autoCattlogV2_sampleCrops_withGTLabels/"

    inferenceEngine.setCattlog(tracksPath_train=tracksPath_train, filterFnName='trackPtsFilter_byProximityOfRBboxToFrameCenter', inclusionPercentage_top=0.2, cattlogDirPath=cattlogDirPath)

    # infer from image frame
    imgPath = "../data/sampleImages/sample1.png"
    frame = cv2.imread(imgPath)
    inferenceEngine.inferFromFrame(frame)

    # infer from video
    # vidPath = "../data/sampleCutVideos/cam24_2022-06-08_05-29-20_6131_76219_76700.avi"
    # inferenceEngine.inferFromVideo_multiInstances(vidPath=vidPath, frameFreq=3, outRootDir="../outputs/ACID_inference_outputs/", standalone=True)

    # infer on video list
    # vidPathsList = glob.glob("../data/sampleCutVideos/*.avi")[:2]  # just taking first 2 videos for testing
    # vidPathsList.sort()

    # inferenceEngine.functionAsAutoCattlogger()
    # inferenceEngine.inferFromVideosList_multiInstances(vidPathsList=vidPathsList, frameFreq=3, videosOutDir="../outputs/ACID_inference_onVideoList_outputs/")


    