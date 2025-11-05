'''
Author: Manu Ramesh

This file is supposed to have functions for color correction/ white or black point compensation.
This is for Project GoVarna.

Here we try to do black/white point correction to improve binarization results.

Helpful link: https://www.youtube.com/watch?v=Z0-iM37wseI
'''

import sys
sys.path.insert(0, '../../') #top level dir
import cv2, torch, pdb, os, pickle
import numpy as np, yaml
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog


from autoCattlogger._autoCattloggerBase import _AutoCattloggerBase

from matplotlib import pyplot as plt
from tqdm import tqdm

#For converting sRGB to linear sRGB and back (gamma expansion and compression)
from dependencies.opencv_srgb_gamma.srgb import to_linear as srgb2linear
from dependencies.opencv_srgb_gamma.srgb import from_linear as linear2srgb

from matplotlib import pyplot as plt

import threading
import multiprocessing as mp
#import multiprocess as mp
import psutil #for checing cpu affininty - for multi processing

from scipy.ndimage import gaussian_filter as gaussian_filter
import warnings

from shapely.geometry import Polygon #for finding IOUs of cows

class CowColorCorrector(_AutoCattloggerBase):

    '''
    This class has many many functions. I apologize for the mess. 
    I was trying out many things.
    But one can use any of these functions in case they need them.

    I have marked the function that I have finally used in the BMLP paper and papers beyond those.
    The various run demo functions each show an example of how to use the functions in this class.    
    '''


    def __init__(self, cfg=None, patchSize=1, printDebugInfoToScreen=True, **kwargs):
        #super(CowColorCorrector, self).__init__(cfg, bitVectorCattlogPath="../Eidetic Cattle Recognition/Models/bit_vector_cattlogs/cowDataMatDict_KP_aligned_june_08_2022.p", cattlogDirPath = "../../Outputs/top_view_KP_aligned_cattlog_june_08_2022/", useHandCraftedKPRuleLimits = True, datasetKpStatsDictPath="../Eidetic Cattle Recognition/output/statOutputs/datasetKpStatsDict_kp_dataset_v4_train.p", affectedKPsDictPath = "../Eidetic Cattle Recognition/output/statOutputs/datasetKpStatsDict_kp_dataset_v4_train_affectedKPs.yml", cowAspRatio_ulim = None, saveHardExamples=False, hardExamplesSavePath="./output/hardExamples/", saveEasyExamples=False, easyExamplesSavePath="./output/easyExamples/", printDebugInfoToScreen=False)

        if cfg is not None:
            
            super(CowColorCorrector, self).__init__(cfg, printDebugInfoToScreen=printDebugInfoToScreen, **kwargs)

            # self.cfg = cfg
            # self.predictor = DefaultPredictor(self.cfg)  # keypoint and mask predictors from Detectron2 - this might use GPU0 only by default.

            self.patchSize = patchSize #are we going to use this?
            self.printDebugInfoToScreen = printDebugInfoToScreen


    def computePatchWiseBlackAndWhitePointsOfFrame(self, frame, mask, patchSize=1):
        '''
        Divides the image into patches of size (patchSize X patchSize) and computes black points and white points for each patch.
        The black points (and white points) are computed only in those patches in which the majority of pixels are True(1 or 255) in the mask.
        :param frame: color image frame (3 channel numpy array)
        :param mask: binary image of same size as the image (1 channel numpy array)
        :param patchSize: size of the image patch. Patch size must divide frameH and frameW completely.
        :return:
        '''


        frameH, frameW, _ = frame.shape
        assert frameH%patchSize==0 and frameW%patchSize==0

        frameSafe = frame.copy()

        binaryMask = mask//np.max(mask)

        blackPointImage = np.zeros_like(frame) #default value should be 0,0,0 black
        whitePointImage = np.ones_like(frame) * 255  # default value should be 255,255,255 white

        for i in range(0,frameH,patchSize):
            for j in range(0,frameW,patchSize):
                binMaskPatch = binaryMask[i:i+patchSize,j:j+patchSize]
                framePatch = frame[i:i+patchSize,j:j+patchSize,:]

                if np.sum(binMaskPatch) >= (patchSize**2)/2: #if the number of white pixels in the mask are greater than the number of black pixels in the mask

                    blackPointImage[i:i+patchSize,j:j+patchSize,:] = np.min(framePatch, axis=(0,1))
                    whitePointImage[i:i + patchSize, j:j + patchSize, :] = np.max(framePatch, axis=(0, 1))
                    #pdb.set_trace()

        if self.printDebugInfoToScreen: print(f"Black point image = {blackPointImage}")
        if self.printDebugInfoToScreen: print(f"White point image = {whitePointImage}")

        #cv2.imshow('blackpointImg', cv2.resize(blackPointImage, (1280,720)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # cv2.imshow('whitepointImg', cv2.resize(whitePointImage, (1280,720)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return blackPointImage, whitePointImage
    
    ########################################################################################################################
    # def extendBlackPointImg_normalizedGaussian(self, cBlkPtImg, outDir='./', kernelSize=301, saveImg=True):
    # Make this static method to avoid need to instantiate the class just to use this function.
    # This will save time and memory - avoids creating Detectron2 predictor object on GPU/RAM unnecessarily.
    @staticmethod
    def extendBlackPointImg_normalizedGaussian(cBlkPtImg, outDir='./', kernelSize=301, saveImg=True):
        '''
        # https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image
        
        Used in BMLP Paper and any paper after that including the AutoCattloger paper.
        This uses normalized convolution with a Gaussian kernel to extend the black point image.

        :param cBlkPtImg: complete black point image (the image to be extended)
        :param kernelSize: size of the kernel. Must be odd number.
        :param saveImg: whether to save the output image or not. Default is True.
        :return: extended image
        
        '''

        os.makedirs(outDir, exist_ok=True)

        # gamma expansion
        cBlkPtImg = srgb2linear(cBlkPtImg)
        
        #compute the mask
        mask = cv2.cvtColor(cBlkPtImg, cv2.COLOR_BGR2GRAY)
        mask[mask!=0] = 1.0
        mask = mask[...,np.newaxis]

        invMask = np.where(mask==1, 0, 1)

        #pdb.set_trace()

        #cv2.imshow('mask', (mask*255).astype(np.uint8))
        #cv2.imshow('invmask', (invMask*255).astype(np.uint8))
        #cv2.waitKey(0)
        
        #https://stackoverflow.com/questions/60798600/why-scipy-ndimage-gaussian-filter-doesnt-have-a-kernel-size
        sigma = (kernelSize-1)/6 #=50 for kernelSize 301
        truncateVal= 3 #always #=(kernelSize-1)/(2*sigma) #truncates to these many sigmas = 3 on each side
        #kernel size is 300 or 301, idk. Should be 301 for odd. IDK how it calculates.

        blurredImg  =  gaussian_filter(cBlkPtImg, sigma = (sigma,sigma,0), truncate=truncateVal, mode='constant', cval=0)    #, axes=(0,1)) - this keyword is not supported in this version of scipy
        #cv2.imshow('blurredImg', np.round(linear2srgb(blurredImg)).astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        weights = gaussian_filter(mask, sigma = (sigma,sigma,0), truncate=truncateVal, mode='constant', cval=0)    #, axes=(0,1)) - this keyword is not supported in this version of scipy
        
        blurredImg /= weights+1e-10 #normalizing with weights only from the seen region

        extendedImg = invMask*blurredImg+cBlkPtImg
        extendedImg = linear2srgb(extendedImg)
        extendedImg = np.round(extendedImg).astype(np.uint8)

        #pdb.set_trace()

        #cv2.imshow('extendedImg', extendedImg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if saveImg:
            cv2.imwrite(f'{outDir}/extendedBlkPtImg_normConvGaussian_kernelSize-{kernelSize}.bmp', extendedImg)  # use this for all processing
            cv2.imwrite(f'{outDir}/extendedBlkPtImg_normConvGaussian_kernelSize-{kernelSize}.png', extendedImg)  # for reading on iPad
        
        return extendedImg
   
    
    def computePatchWiseBlackPointsFromVideos(self, vidPathsList, outImgDir=None, patchSize=1, frameFreq=1, computeBlkPtImg=True, computeWhtPtImg=True, filterFrames=False):
        '''
        Takes in a list of video (with a completely black cow) and computes one black point image for the all frames of all videos.
        This must detect the cow and compute black points in patches where the cow is present in each frame,
        then, combine these black points across frames into a single blackpoint image.

        Pass only one video if you need black/white points computed from a single video.

        :param vidPathsList: list of paths to videos of completely black cows
        :param outImgDir: Make sure that out image is saved as in an uncompressed format.
        :param patchSize:
        :param frameFreq: you pick one in frameFreq frames for computing black points
        :param filterFrames: if True, will median blur frames before processing. This is useful in removing the small white patches on black cows.
        :return:
        '''

        #only one image for all videos put toghether


        completeBlkPtImg = None
        completeWhtPtImg = None

        os.makedirs(outImgDir, exist_ok=True)

        for idx, vidPath in enumerate(vidPathsList):

            cap = cv2.VideoCapture(vidPath)
            ret = True

            frameCount = 0

            while ret:
                ret, frame = cap.read()


                if ret == False:
                    if self.printDebugInfoToScreen: print(f"Breaking out since ret is false")
                    break

                frameSafe = frame.copy()
                frameBlurred = cv2.medianBlur(frame, 51) #11

                frameCount += 1
                if self.printDebugInfoToScreen: print(f"Current frame = {frameCount}")
                # frameSafe = frame.copy()  # so that we do not write anything on this frame

                # remove later
                # if frameCount <129 or frameCount >318: #% 30 != 0:
                if frameCount % frameFreq != 0:
                    if self.printDebugInfoToScreen: print(f"skipping frame")
                    continue
                # for debugging - remove later
                # if frameCount == 55710:
                #    cv2.imwrite(f"frameSample_{frameCount}.jpg", frame)

                outputs = self.predictor(frame)
                # print(f"outputs = \n{outputs}")

                predictions = outputs['instances'].to('cpu')
                boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None

                num_instances = boxes.shape[0]

                if predictions.has("pred_masks"):
                    masks = np.asarray(predictions.pred_masks)
                    # masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
                else:
                    masks = None

                if self.printDebugInfoToScreen: print(f"number of masks in the frame = {masks.shape[0]}")
                if masks.shape[0] > 1:
                    print(f"Found more than one mask in frame. Write code later to handle this case.")
                    #pdb.set_trace()

                for maskNumber in range(masks.shape[0]):

                    mask_img = masks[maskNumber, :, :].astype(np.uint8)  # *255
                    _, mask_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)

                    #if filterFrames:
                    #    overlap = cv2.bitwise_and(frameBlurred, frameBlurred, mask=mask_img)
                    #    print(f"Using Blurred Images")
                    #else:
                    #    overlap = cv2.bitwise_and(frameSafe, frameSafe, mask=mask_img)

                    #cv2.imshow('MaskImg', cv2.resize(mask_img, (1280, 720)))
                    #cv2.imshow('OverlapImg', cv2.resize(overlap, (1280, 720)))
                    #cv2.waitKey(30)

                    #assuming that there is only one mask in the frame

                    if filterFrames:
                        blackPointImg, whitePointImg = self.computePatchWiseBlackAndWhitePointsOfFrame(frame=frameBlurred, mask=mask_img, patchSize=patchSize)
                    else:
                        blackPointImg, whitePointImg = self.computePatchWiseBlackAndWhitePointsOfFrame(frame=frame, mask=mask_img, patchSize=patchSize)


                    if computeBlkPtImg:
                        if completeBlkPtImg is None:
                            completeBlkPtImg = blackPointImg
                        else:
                            completeBlkPtImg = np.maximum(completeBlkPtImg, blackPointImg) #np.maximum computes elementwise maximum and returns the maximum array

                    if computeWhtPtImg:
                        if completeWhtPtImg is None:
                            completeWhtPtImg = whitePointImg
                        else:
                            completeWhtPtImg = np.minimum(completeWhtPtImg, whitePointImg) #np.minimum computes elementwise minimum and returns the minimum array

            cap.release()
            if computeBlkPtImg: cv2.imwrite(f'{outImgDir}/completeBlackPointImage_patchSize{patchSize}_after{idx+1}-videos.bmp', completeBlkPtImg)
            if computeWhtPtImg: cv2.imwrite(f'{outImgDir}/completeWhitePointImage_patchSize{patchSize}_after{idx+1}-videos.bmp', completeWhtPtImg)

        if outImgDir is not None:
            #pdb.set_trace()
            if computeBlkPtImg: cv2.imwrite(f'{outImgDir}/completeBlackPointImage_patchSize{patchSize}.bmp', completeBlkPtImg)
            if computeWhtPtImg: cv2.imwrite(f'{outImgDir}/completeWhitePointImage_patchSize{patchSize}.bmp', completeWhtPtImg)

        return completeBlkPtImg, completeWhtPtImg

    def computePatchWiseBlackPointsFromTrackPoint(self, frame, rotatedBBOX_locs=None, trackPoint=None, patchSize=1, predictor=None, filterFrames=False):
        '''
        For ACv2.
        Takes in a frame and a rotated bbox corner points (or gets the rotatedBBOX pts from the track point).
        Runs a cow mask detector on the frame and computes the illumination map (blackPoint image) using the mask that falls within the rotatedBBOX.

        :param frame: the frame to be processed. This is the frame from the video.
        :param rotatedBBOX_locs: the rotated bbox corner points. This is a 4x2 array of points. If None, will get the points from the trackPoint.
        :param trackPoint: the track point from the video. This is a dictionary with keys 'frameNumber', 'rotatedBBOX_locs', 'blackPointImg', and more. This parameter is required.
        :param patchSize: the patch size to be used for computing the black point image. This is the size of the patch in pixels. Default is 1 - 1 was used for BMLP paper and other papers after that.
        :param predictor: the cow mask detector. This is a Detectron2 predictor object. (You can pass some other mask detector too with the same interface.)
        :param filterFrames: if True, will median blur frames before processing. This is useful in removing the small white patches on black cows.
        '''

        if rotatedBBOX_locs is None:
            rotatedBBOX_locs = trackPoint['rotatedBBOX_locs']

        #Get all the cow masks from the frame
        if predictor is not None:
            outputs = predictor(frame) #just to make it work with multiprocessing
        else:
            outputs = self.predictor(frame)
        # print(f"outputs = \n{outputs}")

        predictions = outputs['instances'].to('cpu')
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None

        num_instances = boxes.shape[0]

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            # masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self.printDebugInfoToScreen: print(f"number of masks in the frame = {masks.shape[0]}")
         #pdb.set_trace()

        list_of_rotatedBBoxes = []


        for maskNumber in range(masks.shape[0]):
            
            #print(f"Current Mask = {maskNumber}")

            mask_img = masks[maskNumber, :, :].astype(np.uint8)  # *255
            _, mask_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 1:
                #logging.debug(f"More than one contour detected. Not Breaking. Using largest contour.")
                #print(f"More than one contour detected. Not Breaking. Using largest contour.")
                contours = sorted(contours, key=cv2.contourArea)[::-1]  # sort in descending order

            if len(contours) == 0:
                warnings.warn("No contours found in the mask. Make sure to use the same mask detector model as used during track generation.")
                continue
            
            cnt = contours[0]

            #Compute the rotated bbox pts for the mask
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # 4 x 2 np array [[xA, yA], [xB, yB], [xC, yC], [xD, yD]
            box = np.intp(box) #np.int0(box)  # converting to int64
            list_of_rotatedBBoxes.append((box, mask_img))


        #here, we set the default values to the black and white point images.
        blackPointImg = np.zeros_like(frame)
        whitePointImg = np.ones_like(frame) * 255

        
        if len(list_of_rotatedBBoxes) == 0:
            #if self.printDebugInfoToScreen: print(f"No masks detected in the frame {trackPoint['frameNumber']}.")
            if True: print(f"No masks detected in the frame {trackPoint['frameNumber']}.")

            #this will return the default values of black and white point images.
            pass


        else:

            #find the mask that coincides with the given rotatedBBOX
            #list_of_rotatedBBoxes.sort(key=lambda x: Polygon(rotatedBBOX_locs).intersection(Polygon(x[0]).area), reverse=True) #sort in descending order of intersesction area
            IOU_list = [Polygon(rotatedBBOX_locs).intersection(Polygon(x[0])).area/Polygon(rotatedBBOX_locs).union(Polygon(x[0])).area for x in list_of_rotatedBBoxes]
            #print(f"Intersection Areas = {IOU_list}")

            #try:
            maxIOU_idx = np.argmax(IOU_list)




            if IOU_list[maxIOU_idx] < 0.1:
                print(f"\nWarning! Very low IOU found between the rotatedBBOX and the masks in frame {trackPoint['frameNumber']}.\nIOU = {IOU_list[maxIOU_idx]}. This could be from the wrong cow instance. Skipping the instance.\n")
                
                #default values of black and white point images will be returned.
                #this condition is added to handle cases where the maks for the black cow is not found, but a mask for any other cow is found. Such a mask form a different track should not be used to compute the black and white points.
                
            else:

                if IOU_list[maxIOU_idx] < 0.8:
                    print(f"\nWarning! Low IOU found between the rotatedBBOX and the masks. IOU = {IOU_list[maxIOU_idx]}\n")
                    #pdb.set_trace()

                requiredMaskImg = list_of_rotatedBBoxes[maxIOU_idx][1]
                
                if filterFrames:
                    frameBlurred = cv2.medianBlur(frame, 51) #11
                    blackPointImg, whitePointImg = self.computePatchWiseBlackAndWhitePointsOfFrame(frame=frameBlurred, mask=requiredMaskImg, patchSize=patchSize)
                
                else:
                    blackPointImg, whitePointImg = self.computePatchWiseBlackAndWhitePointsOfFrame(frame=frame, mask=requiredMaskImg, patchSize=patchSize)


        return blackPointImg, whitePointImg


    def computePatchWiseBlackPointsFromTrack(self, tracksList, srcVideosDir, gt_label=None, trackId=None, outImgDir=None, patchSize=1, filterFrames=False, computeBlkPtImg=True, computeWhtPtImg=True):
        '''
        For ACv2.
        Takes in a list of tracks and the gt_label/track id of the black cow.
        Computes the black point image from the track using the videos in the srcVideosDir.
        Processes one frame (from one trackPoint) at a time.

        This is a single process version of the function. The multi process version is computePatchWiseBlackPointsFromTrack_multiProc.
        Use this to debug any issues with the multi process version.
        I don't use this as it is slower than the multiprocessing version.
        Use this if you have issues with the multi process version.

        :param tracksList: list of tracks from the track file. This is a list of dictionaries with keys 'trackId', 'gt_label', 'trackPoints', and more.
        :param srcVideosDir: This is the directory where the videos from which the tracks were generated are stored.
        :param gt_label: the gt label (CowID) of the black cow. This is a string. If None, will use trackId to find the track.
        :param trackId: the track id of the black cow. This is a string. If None, will use gt_label to find the track.
        :param outImgDir: the directory where the intermediate outputs of extended black point image will be saved. Use this to debug. If None, will not save the images.
        :param patchSize: the patch size to be used for computing the black point image. This is the size of the patch in pixels. Default is 1 - 1 was used for BMLP paper and other papers after that.
        :param filterFrames: if True, will median blur frames before processing. This is useful in removing the small white patches on black cows.
        :param computeBlkPtImg: if True, will compute the black point image. Default is True.
        :param computeWhtPtImg: if True, will compute the white point image. Default is True.

        :return: the black point image and the white point image. These are the images that are used to correct the illumination in the video.
        '''

        assert trackId is not None or gt_label is not None, "Please provide either trackID or gt_label."

        track = []

        pdb.set_trace()

        if trackId is not None:
            track = [track for track in tracksList if track['trackId'] == trackId][0] #index out of bounds here means that the trackId is not found in tracksList, trackId should be INT
        elif gt_label is not None:
            track = [track for track in tracksList if track['gt_label'] == gt_label][0] #index out of bounds here means that the gt_label is not found in tracksList, gt_label should be STRING
        
        if len(track) == 0:
            print(f"No track found for trackId = {trackId} and gt_label = {gt_label}")
            #logging.debug(f"No track found for trackId = {trackId} and gt_label = {gt_label}")
            return None, None

        completeBlkPtImg = None
        completeWhtPtImg = None

        if outImgDir is not None:
            os.makedirs(outImgDir, exist_ok=True)

        currentVideoName = None
        cap = None; ret = True

        #delete later
        #track['trackPoints'] = track['trackPoints'][75:] #for debugging

        for trackPoint in track['trackPoints']:

            if (currentVideoName is None) or (currentVideoName != trackPoint['videoName']):
                #Do not recraete video capture object if the video is the same. This should eliminate reinitilaization times.
                currentVideoName = trackPoint['videoName']
                vidPath = f"{srcVideosDir}/{trackPoint['videoName']}"
                
                if cap is not None:
                    cap.release() #release the previous video capture object

                cap = cv2.VideoCapture(vidPath)
                ret = True

            
            frameNumber = trackPoint['frameNumber'] -1 #+1 # -1 is correct. I verified by printing frame numbers from track points (on the frame) and video with frame numbers in them.

            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber) #make sure that the frame rate of the videos from the Ubiquity cameras are corrected to 30fps before running this code
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frameNumber} from video {vidPath}")
                #continue
                break
            

            #cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
            #cv2.waitKey(1)

            blackPointImg, whitePointImg = self.computePatchWiseBlackPointsFromTrackPoint(frame, rotatedBBOX_locs=trackPoint['rotatedBBOX_locs'], trackPoint=trackPoint, patchSize=patchSize, filterFrames=filterFrames)

            if computeBlkPtImg:
                if completeBlkPtImg is None:
                    completeBlkPtImg = blackPointImg
                else:
                    completeBlkPtImg = np.maximum(completeBlkPtImg, blackPointImg)

            if computeWhtPtImg:
                if completeWhtPtImg is None:
                    completeWhtPtImg = whitePointImg
                else:
                    completeWhtPtImg = np.minimum(completeWhtPtImg, whitePointImg)

            if outImgDir is not None:
                if computeBlkPtImg: cv2.imwrite(f'{outImgDir}/completeBlackPointImage_patchSize{patchSize}_after{trackPoint["frameNumber"]}-frames.jpg', completeBlkPtImg) #Saving just to visualize. Saving as jpg to save space.
                if computeWhtPtImg: cv2.imwrite(f'{outImgDir}/completeWhitePointImage_patchSize{patchSize}_after{trackPoint["frameNumber"]}-frames.jpg', completeWhtPtImg) #Saving just to visualize. Saving as jpg to save space.

        if cap is not None:
            cap.release()

        return completeBlkPtImg, completeWhtPtImg

    def computePatchWiseBlackPointsFromTrack_multiProc(self, tracksList, srcVideosDir, gt_label=None, trackId=None, outImgDir=None, patchSize=1, filterFrames=False, computeBlkPtImg=True, computeWhtPtImg=True, numWorkers=9):
        '''
        For ACv2.
        Takes in a list of tracks and the gt_label/track id of the black cow.
        Computes the black point image from the track using the videos in the srcVideosDir.
        It is a multiprocessing approach.
        Each process featches the frame corresponding to a trackPoint and computes the black point image for that frame.
        After all processes are done, the black point images are combined into a single black point image.
        (The single process version is computePatchWiseBlackPointsFromTrack.)

        I've noticed a problem with seeking in H264 encoded videos. So, seeking the frame to the required frame number will give you a frame in the viscinity. 
        The illumination maps for the AutoCattlogger paper were generated using this method.
        But, it does not matter if we get a few frames here and there because we accumulate all pixels from the mask in the end. 

        In its raw form, this approach's memory usage scales linearly with the number of frames in the track video. 
        (Roughly 6.3MB per frame - 1920x1080x3 bytes per frame - 1.8GB for 300 frames or 10 seconds)
        So, i wanted to implement a shared memory approach that would make the subprocess update a common illuinination map.
        However, the way it is implemented now is the best way that I could do it. (Which is to process the trackpoints in chunks in a loop.)
        I tried to use multiprocessing.shared_memory.SharedMeory, multiprocessing.Array, multiprocessing.Manager etc.
        However, the shared memory approaches resulted in one or the other roadblocks.

        :param tracksList: list of tracks from the track file. This is a list of dictionaries with keys 'trackId', 'gt_label', 'trackPoints', and more.
        :param srcVideosDir: This is the directory where the videos from which the tracks were generated are stored.
        :param gt_label: the gt label (CowID) of the black cow. This is a string. If None, will use trackId to find the track.
        :param trackId: the track id of the black cow. This is a string. If None, will use gt_label to find the track.
        :param outImgDir: the directory where the intermediate outputs of extended black point image will be saved. Use this to debug. If None, will not save the images.
        :param patchSize: the patch size to be used for computing the black point image. This is the size of the patch in pixels. Default is 1 - 1 was used for BMLP paper and other papers after that.
        :param filterFrames: if True, will median blur frames before processing. This is useful in removing the small white patches on black cows.
        :param computeBlkPtImg: if True, will compute the black point image. Default is True.
        :param computeWhtPtImg: if True, will compute the white point image. Default is True.
        '''

        assert trackId is not None or gt_label is not None, "Please provide either trackID or gt_label."

        try:
            torch.multiprocessing.set_start_method('spawn') #this is needed for multiprocessing to work with pytorch & GPU, and needs to be run only once.
        except:
            pass #this is needed to avoid the error: "RuntimeError: context has already been set". Running the torch.multiprocessing.set_start_method('spawn') function more than once results in this error.

        track = []

        if trackId is not None:
            track = [track for track in tracksList if track['trackId'] == trackId][0]
        elif gt_label is not None:
            track = [track for track in tracksList if track['gt_label'] == gt_label][0]
        
        
        if len(track) == 0:
            print(f"No track found for trackId = {trackId} and gt_label = {gt_label}")
            #logging.debug(f"No track found for trackId = {trackId} and gt_label = {gt_label}")
            return None, None

        completeBlkPtImg = None
        completeWhtPtImg = None

        if outImgDir is not None:
            os.makedirs(outImgDir, exist_ok=True)


        
        #track['trackPoints'] = track['trackPoints'][:6] #for debugging

        processFn = self.computePatchWiseBlackPointsFromTrackPoint
        predictor = self.predictor

        #PROCESS FRAMES IN BATCHES TO SAVE MEMORY
        frameComputeBufferSize = 1000 #process these many frames at a time, change this depending on the amount of RAM available on your mahcine

        for i in range(0, len(track['trackPoints']), frameComputeBufferSize): #process 1000 frames at a time

            #The memory occupied by the results list will be reused in every iteration. This reduces the memory usage.
            #however, every iteration involves reloading copies of the mask predictor model to the GPU. So, do not reeduce the frameComputeBufferSize too much.

            trackPtsSublist = track['trackPoints'][i:i+frameComputeBufferSize]

            paramsList = [(srcVideosDir, trackPoint, processFn, patchSize, predictor, filterFrames) for trackPoint in  trackPtsSublist]
            pool = mp.Pool(processes=min(numWorkers, os.cpu_count())) 

            #results = pool.map(getBlackWhitePointsFromTrackPoint_forMultiProc, paramsList)
            #Progress bar for GPU: https://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-imap-unordered-call
            
            results = list(tqdm(pool.imap_unordered(getBlackWhitePointsFromTrackPoint_forMultiProc, paramsList), total=len(paramsList)))

            pool.close()
            pool.join()

            for blackPointImg, whitePointImg in results:

                if computeBlkPtImg:
                    if completeBlkPtImg is None:
                        completeBlkPtImg = blackPointImg
                    else:
                        completeBlkPtImg = np.maximum(completeBlkPtImg, blackPointImg)

                if computeWhtPtImg:
                    if completeWhtPtImg is None:
                        completeWhtPtImg = whitePointImg
                    else:
                        completeWhtPtImg = np.minimum(completeWhtPtImg, whitePointImg)

                #if outImgDir is not None:
                #    if computeBlkPtImg: cv2.imwrite(f'{outImgDir}/completeBlackPointImage_patchSize{patchSize}_after{trackPoint["frameNumber"]}-frames.bmp', completeBlkPtImg)
                #    if computeWhtPtImg: cv2.imwrite(f'{outImgDir}/completeWhitePointImage_patchSize{patchSize}_after{trackPoint["frameNumber"]}-frames.bmp', completeWhtPtImg)

        return completeBlkPtImg, completeWhtPtImg


    def blackPointCorrectVideo(self, vidPath, blkPtImg, outDir='./', frameFreq=1, patchSize=1):
        '''
        Takes in a video (path) and a black point correction image.
        Corrects black point patch wise on every frame in the video.

        Useful to visualize color correction.

        :param vidPath:
        :param blkPtImg:
        :param outDir: path to output direcotry
        :return:
        '''

        cap = cv2.VideoCapture(vidPath)
        ret = True
        frameCount = 0

        frameW = 1920; frameH = 1080

        out = cv2.VideoWriter(f'{outDir}/blkPtCorrectedVideo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frameW, frameH))

        print(f"Trying to black point correct video at {vidPath}")

        while ret:
            ret, frame = cap.read()

            if ret == False:
                if self.printDebugInfoToScreen: print(f"Breaking out since ret is false")
                break

            frameSafe = frame.copy()

            frameCount += 1
            if self.printDebugInfoToScreen: print(f"Current frame = {frameCount}")
            # frameSafe = frame.copy()  # so that we do not write anything on this frame

            # remove later
            # if frameCount <129 or frameCount >318: #% 30 != 0:
            if frameCount % frameFreq != 0:
                if self.printDebugInfoToScreen: print(f"skipping frame")
                continue
            # for debugging - remove later
            # if frameCount == 55710:
            #    cv2.imwrite(f"frameSample_{frameCount}.jpg", frame)

            #blkPtCorrectedFrame = self.correctPatchWiseBlackPointsOnFrame(frame=frame, blackPointImg=blkPtImg, patchSize=patchSize)
            blkPtCorrectedFrame = self.correctPatchWiseBlackPointsOnFramePS1(frame=frame, blackPointImg=blkPtImg, patchSize=1) #fixed patch size 1

            #cv2.imshow('blackPointCorrectedFrame', cv2.resize(blkPtCorrectedFrame, (1280,720)))
            #cv2.waitKey(30)

            out.write(blkPtCorrectedFrame)

        out.release()
        cap.release()


    global getBlackWhitePointsFromTrackPoint_forMultiProc
    def getBlackWhitePointsFromTrackPoint_forMultiProc(args):
        '''
        The function has to be outisde the class for it to be used in the multiprocessing pool. (Python's rules.)
        The function takes in a tuple of arguments and returns the black and white point images.
        It is just an extra step to make the code work with multiprocessing.
        
        I wrote it as a wrapper that takes in both the function and the arugments that are to be passed to the function as a tuple.
        It then calls the function with the arguments and returns the result.

        :params args: tuple (srcVideosDir, trackPoint, processFn, patchSize, predictor, filterFrames)
        Check the function computePatchWiseBlackPointsFromTrack_multiProc() for more details about these arguments.

        :return: blackPointImg, whitePointImg
        '''

        #srcVideosDir, trackPoint = args
        srcVideosDir, trackPoint, processFn, patchSize, predictor, filterFrames = args
            
        vidPath = f"{srcVideosDir}/{trackPoint['videoName']}"
        cap = cv2.VideoCapture(vidPath)
        ret = True
        frameNumber = trackPoint['frameNumber'] -1 #-1 #+1 #+1 #Not -1,0 as many frames have low IOU with masks and rotatedBBOXes. +1 has no instances with low IOUs # -1 is correct. I verified by printing frame numbers from tracks on video frames that have the actual frame numbers on them.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber) #make sure that the frame rate of the videos from the Ubiquity cameras are corrected to 30fps before running this code
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frameNumber} from video {vidPath}")
            #continue
            return None, None
        
        #cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
        #cv2.waitKey(1)
        #blackPointImg, whitePointImg = self.computePatchWiseBlackPointsFromTrackPoint(frame, rotatedBBOX_locs=trackPoint['rotatedBBOX_locs'], trackPoint=None, patchSize=patchSize)
        
        #try:
        blackPointImg, whitePointImg = processFn(frame, rotatedBBOX_locs=trackPoint['rotatedBBOX_locs'], trackPoint=trackPoint, patchSize=patchSize, filterFrames=filterFrames, predictor=predictor)
        #except:
        #    blackPointImg, whitePointImg = None, None
        #    print(f"Error in processing frame {frameNumber} from video {vidPath}")
        #    
        #    cv2.imwrite(f"./errorFrame_{frameNumber}.jpg", frame)
        
        cap.release()
        return blackPointImg, whitePointImg



def build_CowColorCorrector_from_config(ac_config=None, configPath=None):
    '''
    Utility function to build CowColorCorrector object from a config file.
    :param configPath: path to the config file (YAML format)
    :return: CowColorCorrector object
    '''

    topLevelDirPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # to offset all relative paths in the config file

    assert (ac_config is not None) or (configPath is not None), "Either ac_config or configPath must be provided."

    if ac_config is None:
        ac_config = yaml.safe_load(open(configPath, 'r'))

    detectron2_device = int(ac_config['detectron2']['device']) if torch.cuda.is_available() else 'cpu' #= 0 if only 1 gpu is available

    # get detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(topLevelDirPath + '/' + ac_config['detectron2']['model_config_path']) #must adjust path to start from top level directory
    cfg.MODEL.DEVICE = detectron2_device
    cfg.MODEL.WEIGHTS = topLevelDirPath + '/' + ac_config['detectron2']['weights_path'] #must adjust path to start from top level directory
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ac_config['detectron2']['MODEL_ROI_HEADS_SCORE_THRESH_TEST']

    #mmpose config
    useMMPOSE = ac_config['mmpose']['useMMPOSE'] #should be True
    mmpose_poseConfigPath = topLevelDirPath + '/' + ac_config['mmpose']['mmpose_poseConfigPath'] #must adjust path to start from top level directory
    mmpose_modelWeightsPath = topLevelDirPath + '/' + ac_config['mmpose']['mmpose_modelWeightsPath'] #must adjust path to start from top level directory
    mmpose_device = ac_config['mmpose']['device'] if torch.cuda.is_available() else 'cpu'

    # pdb.set_trace(header="Building AutoCattlogger object from config")

    # build AutoCattlogger object
    cccObject = CowColorCorrector(cfg=cfg,
                              affectedKPsDictPath=topLevelDirPath + '/' + ac_config['AutoCattlogger']['affectedKPsDictPath'],
                              datasetKpStatsDictPath=topLevelDirPath + '/' + ac_config['AutoCattlogger']['datasetKpStatsDictPath'],
                              cowAspRatio_ulim=ac_config['AutoCattlogger'].get('cowAspRatio_ulim', 0.53),
                              cowAreaRatio_llim=ac_config['AutoCattlogger'].get('cowAreaRatio_llim', 0.042),
                              saveHardExamples=ac_config['AutoCattlogger'].get('saveHardExamples', False),
                              saveEasyExamples=ac_config['AutoCattlogger'].get('saveEasyExamples', False),
                              printDebugInfoToScreen=ac_config['AutoCattlogger'].get('printDebugInfoToScreen', False),
                              keypointCorrectionOn=ac_config['AutoCattlogger'].get('keypointCorrectionOn', True),
                              colorCorrectionOn=ac_config['AutoCattlogger'].get('colorCorrectionOn', True),
                              blackPointImgPth=topLevelDirPath + '/' + ac_config['AutoCattlogger'].get('blackPointImgPth', None) if ac_config['AutoCattlogger'].get('blackPointImgPth', None) is not None else None,
                              useMMPOSE=useMMPOSE,
                              mmpose_poseConfigPath=mmpose_poseConfigPath,
                              externalCC_fn=None, #ac_config['AutoCattlogger']['externalCC_fn'], #force None for now
                              threshVal_CC=None if ac_config['AutoCattlogger'].get('threshVal_CC', "None") == "None" else int(ac_config['AutoCattlogger']['threshVal_CC']),
                              mmpose_modelWeightsPath=mmpose_modelWeightsPath,
                              mmpose_device=mmpose_device)


    return cccObject



if __name__ == "__main__":
    
    cccObject = build_CowColorCorrector_from_config(configPath='../../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml')
    
    # tracksList = pickle.load(open("../../outputs/temp/tracks_withGTLabels.pkl", 'rb'))
    tracksList = pickle.load(open("../../outputs/AC_outputs/requiredTracks.pkl", 'rb'))
    srcVideosDir = "../../data/sampleVideos/"
    gt_label = "6116"
    trackID = 1137 #must be int
    outImgDir = "../../outputs/temp/blackPointImages_debug/"
    
    # cccObject.computePatchWiseBlackPointsFromTrack_multiProc(tracksList=tracksList, srcVideosDir=srcVideosDir, gt_label=gt_label, outImgDir=outImgDir, patchSize=1, filterFrames=True)
    # cccObject.computePatchWiseBlackPointsFromTrack(tracksList=tracksList, srcVideosDir=srcVideosDir, gt_label=gt_label, outImgDir=outImgDir, patchSize=1, filterFrames=True)
    
    # cccObject.computePatchWiseBlackPointsFromTrack(tracksList=tracksList, srcVideosDir=srcVideosDir, trackId=trackID, outImgDir=outImgDir, patchSize=1, filterFrames=True)
    completeBlkPtImg, completeWhtPtImg = cccObject.computePatchWiseBlackPointsFromTrack_multiProc(tracksList=tracksList, srcVideosDir=srcVideosDir, trackId=trackID, outImgDir=outImgDir, patchSize=1, filterFrames=True)
    cv2.imwrite(f"{outImgDir}/finalCompleteBlackPointImage_trackID{trackID}.png", completeBlkPtImg)
    # cv2.imwrite(f"{outImgDir}/finalCompleteWhitePointImage_trackID{trackID}.png", completeWhtPtImg)

    CowColorCorrector.extendBlackPointImg_normalizedGaussian(cBlkPtImg=completeBlkPtImg, outDir=outImgDir, saveImg=True)