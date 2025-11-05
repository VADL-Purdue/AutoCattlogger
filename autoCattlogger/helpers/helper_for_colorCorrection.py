'''
Author: Manu Ramesh

This file is supposed to have functions for color correction/ white or black point compensation.
This is for Project GoVarna.

Here we try to do black/white point correction to improve binarization results.

Helpful link: https://www.youtube.com/watch?v=Z0-iM37wseI


TEMPORARY FILE. TO AVOID CIRCULAR IMPORT ERROR.

'''

import sys
sys.path.insert(0, '../') # top level directory

import cv2, torch, pdb
import numpy as np
#from joblib import Parallel, delayed

#For converting sRGB to linear sRGB and back (gamma expansion and compression)
from dependencies.opencv_srgb_gamma.srgb import to_linear as srgb2linear
from dependencies.opencv_srgb_gamma.srgb import from_linear as linear2srgb

from tqdm import tqdm, trange
import glob, os

#out of class function
'''
Taking function out of class as i need to use this function inside the _AutoCattloggerBase class in the _autoCattloggerBase.py file.
Problem could arise as the class CowColorCorrector is a child of _AutoCattloggerBase and i would have to create an instance of this child inside the parent.
'''
def correctPatchWiseBlackPointsOnFrame(frame, blackPointImg, patchSize=1):
    '''
    Takes in the frame and the black point image and black point compensates in each patch separately.
    If you specify the patch size, make sure you use the same patch size used to create the blackpoint image.
    Otherwise this might not work properly. If you are unsure as to what patch size was used to create the black point image,
    set the patchSize to 1.
    :param frame: If you used bgr frame to compute black point image, use bgr frame here. Using RGB while creating black point image and BGR here will lead to the wrong result.
    :param blackPointImg:
    :param patchSize:
    :return:
    '''

    frameH, frameW, _ = frame.shape
    assert frameH % patchSize == 0 and frameW % patchSize == 0

    frameSafe = frame.copy()
    outImage = np.zeros_like(frame)

    for i in range(0, frameH, patchSize):
        for j in range(0, frameW, patchSize):
            #outPatch = outImage[i:i + patchSize, j:j + patchSize]
            framePatch = frame[i:i + patchSize, j:j + patchSize, :]

            blackPointImgPatch = blackPointImg[i:i + patchSize, j:j + patchSize, :]
            # we could have picked just one point since all points are same. This is just to ensure that in case the patchSizes are not the same when creating black point image and now, we get a better image with patchSize=1.
            blackPoint = np.median(blackPointImgPatch, axis=[0,1])

            outPatch = ((framePatch - blackPoint)/((255-blackPoint)+1e-10)).clip(0,1) * 255
            #does changing outPatch automatically change the outImage?
            outImage[i:i + patchSize, j:j + patchSize] = outPatch

    return outImage

def correctPatchWiseBlackPointsOnFramePS1(frame, blackPointImg, patchSize=1):
    '''
    Takes in the frame and the black point image and black point compensates in each patch separately.
    If you specify the patch size, make sure you use the same patch size used to create the blackpoint image.
    Otherwise this might not work properly. If you are unsure as to what patch size was used to create the black point image,
    set the patchSize to 1.
    :param frame: If you used bgr frame to compute black point image, use bgr frame here. Using RGB while creating black point image and BGR here will lead to the wrong result.
    :param blackPointImg:
    :param patchSize:
    :return:

    works for patch size 1. Faster.
    '''

    #frameH, frameW, _ = frame.shape
    #assert frameH % patchSize == 0 and frameW % patchSize == 0

    #frameSafe = frame.copy()
    #outImage = np.zeros(frame.shape, float)

    # old method -- without gamma correction
    # outImage = ((frame.astype(float) - blackPointImg.astype(float)) / ((255 - blackPointImg.astype(float)) + 1e-10)).clip(0, 1) * 255
    # outImage = outImage.astype(np.uint8)

    #new method, with gamma expansion and recompression -- the correct way
    blackPointImg_lin = srgb2linear(blackPointImg.copy())
    frame_lin = srgb2linear(frame.copy())
    outImage = ((frame_lin - blackPointImg_lin) / ((1.0 - blackPointImg_lin) + 1e-10)).clip(0, 1)
    outImage = linear2srgb(outImage).astype(np.uint8)



    return outImage



########################################################################################################################

########################################################################################################################
#### Some other functions

def colorCorrectVideoList(vidPathsList, colorCorrectionFn, outRootDir, outDirSuffix='', **kwargs):
    '''
    To just see how the color correct videos look.

    :param vidPathsList: List of paths to videos.
    :param colorCorrectionFn: Pass any color correction function here.
    :param outRootDir: Path to output directory
    :param outDirSuffix: suffix for output directory
    :param **kawargs: Pass any arguments that need to be passed to the color correction function h ere.
    '''

    frame_width = 1920; frame_height = 1080

    outDir = f"{outRootDir}/colorCorrectedVideos_{vidPathsList[0].split('/')[-2]}_{outDirSuffix}/"
    os.makedirs(outDir, exist_ok = True)

    pbar = tqdm(vidPathsList, unit='cowVideo')  # tqdm progress bar

    print(f"\nColor correcting cow videos")

    for idx, vidPath in enumerate(pbar): #you enumerate the pbar. if you tqdm(enumerate(vidList)), tqdm will not know how many cows are in the list, so it won't display eta or the progress bar!

        cap = cv2.VideoCapture(vidPath)
        ret = True
        frameCount = 0

        vidName = vidPath.split('/')[-1].split('.')[0] + "_CC" + '.avi'

        #print(f"Out vid = {outDir}/{vidName}")

        # saving video
        out = cv2.VideoWriter(f"{outDir}/{vidName}", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))


        while ret:
            ret, frame = cap.read()
            if ret == False:
                #print(f"Breaking out since ret is false")
                break
            frameCount += 1

            frameCC = colorCorrectionFn(frame, **kwargs)

            out.write(frameCC)

        out.release()
        cap.release()


    print(f"Color correctred videos are in {outDir}")


#########################################################################################


if __name__ == "__main__":

    #vid_paths_list = glob.glob("../../Data/CowTopView/Datasets/vid_dataset_v4/videos_train_set/*.avi") #train set (misnomer alert)
    #vid_paths_list = glob.glob("../../Data/CowTopView/Datasets/vid_dataset_v4/videos_train_set/*2205*.avi") #one cow of train set
    vid_paths_list = glob.glob("../../Data/CowTopView/Datasets/vid_dataset_v4/videos_test_inSet/*.avi")  # test inSet
    #vid_paths_list = glob.glob("../../Data/CowTopView/Datasets/vid_dataset_v4/videos_test_inSet/*.avi")[:3] #test inSet - just a few cows - for quick evaluations

    blackPointImgPth = f'./Models/dairyIllumination_models/extendedBlkPtImg_regionSize301_linearRGB.bmp'  # None #Path to image used for black point correction #new location -- gamma correct version
    blackPointImg = cv2.imread(blackPointImgPth)

    #colorCorrectVideoList(vidPathsList=vid_paths_list, colorCorrectionFn=correctPatchWiseBlackPointsOnFramePS1, outRootDir=f'./output/ColorCorrectedVideos/', outDirSuffix='pwbpc', blackPointImg=blackPointImg) #patch wise black point correction


    ######### INTRINSIC IMAGE DECOMPOSITION - PER PIXEL WHITE BALANCING ###############
    #import color correction function from colorCorrection module and use them here
    #External Color Correction Function
    #from colorCorrection_handler import IID_WB_Handler
    #sys.path.append('./colorCorrectionModules')
    #
    #wb_type ='paper' #diffue
    #
    #if wb_type == 'paper':
    #    cc_serverPort = 65432+100
    #else: #diffues mode
    #    cc_serverPort = 65434+100
    #iidHandler      = IID_WB_Handler(condaEnvName='IntrinsicImageDecomp', wb_type=wb_type, serverPort=cc_serverPort)
    #externalCC_fn   = iidHandler.whiteBalanceFrame #None
    #colorCorrectVideoList(vidPathsList=vid_paths_list, colorCorrectionFn=externalCC_fn, outRootDir=f'./output/ColorCorrectedVideos/', outDirSuffix='iid_paperMode')  # patch wise black point correction

    #trying to see what black point correction does to a white frame
    colorFrame = np.zeros((1080,1920,3), np.uint8)
    colorFrame[:,:,0]=250 #B of BGR
    colorFrame[:,:,1:] = 250
    
    #pdb.set_trace()
    bpc_colorFrame = correctPatchWiseBlackPointsOnFramePS1(frame=colorFrame, blackPointImg=blackPointImg, patchSize=1)
    cv2.imwrite('blackPointCorrectedColorFrame.bmp', bpc_colorFrame)
    cv2.imwrite('blackPointCorrectedColorFrame.png', bpc_colorFrame)
    