'''
Author: Manu Ramesh | VADL | Purdue ECE
This file has code that generates the frames that go into the inference output videos.
In the output frame, the get_cow_matches_frame() inserts
-   the input frame with detection overlays
-   frame numbers
-   predicted cow ID
-   template aligned version of the presented cow, template aligned version of the cow in the cattlog whose id is predicted by the predictor
'''
import pdb

import numpy as np
import cv2

def get_cow_matches_frame(inFrame=[],cropList=[],predIdList=[],cattlogDirPath="", maxCows=4, cowAR = 0.5, frameNumber=None, printDebugInfoToScreen=False):
    """
    Returns a frame that displays
    :param frame: input frame
    :param cropList: list of all cropped cow images
    :param predIdList: list of all predicted ids
    :param maxCows: maximum number of cows that should be displayed on the side.
    :param cattlogDirPath: path to canonical/iconic cow images catalog
    :param frameNumber: to display the frame number from the original video - use later for logging and cross checking

    Note: This code auto scales the input frame based on the max number of cows that are displayed on the side.

    :return: output frame
    """

    outFrameH, outFrameW, outFrameCh = 1080, 1920, 3
    outFrame = np.zeros((outFrameH, outFrameW, outFrameCh), np.uint8)

    predCowLoc = []; iconCowLoc = []
    predCowTextLoc = []; iconCowTextLoc = []
    predCowTextShadowLoc = []; iconCowTextShadowLoc = [] #shadow helps improve readability

    cowH = outFrameH//maxCows
    cowW = int(cowH*cowAR)
    if printDebugInfoToScreen: print(f"outFrame shape = {outFrame.shape}")

    #inframe aspect ratio is assumed to be 16:9
    inFrameOutW = outFrameW - 2*cowW
    inFrameOutW = 16 * int(inFrameOutW/16) #extra space will be used as border on the right
    inFrameOutH = int(9/16 * inFrameOutW)

    if printDebugInfoToScreen: print(f"inframeOutH = {inFrameOutH}, inframeOutW = {inFrameOutW}")

    inFrameLoc = outFrame[(outFrameH-inFrameOutH)//2:(outFrameH-inFrameOutH)//2 + inFrameOutH, :inFrameOutW, :]

    inFrameLoc[:,:,:] = cv2.resize(inFrame, (inFrameOutW, inFrameOutH))

    for i in range(maxCows):
        if printDebugInfoToScreen: print(f"pred locs = {int(i*cowH)}:{int((i+1)*cowH)}, {int(outFrameW-2*cowW)}:{int(outFrameW-1*cowW)}")
        predCowLoc.append(outFrame[int(i*cowH):int((i+1)*cowH), int(outFrameW-2*cowW):int(outFrameW-1*cowW), : ])
        iconCowLoc.append(outFrame[int(i*cowH):int((i+1)*cowH), int(outFrameW-cowW):, :])

        predCowTextLoc.append((int(outFrameW - 2 * cowW), int((i + 1) * cowH - 1)))
        iconCowTextLoc.append((int(outFrameW - cowW), int((i + 1) * cowH - 1)))

        predCowTextShadowLoc.append((int(outFrameW - 2 * cowW + 1), int((i + 1) * cowH - 1 + 1)))
        iconCowTextShadowLoc.append((int(outFrameW - cowW + 1), int((i + 1) * cowH - 1 + 1)))

    for idx in range(min(len(predIdList), maxCows)):
        predId = predIdList[idx]
        iconImg = cv2.imread(f"{cattlogDirPath}/{predId}.jpg")
        iconCowLoc[idx][:,:,:] = cv2.resize(iconImg, (cowW,cowH))
        predCowLoc[idx][:, :, :] = cv2.resize(cropList[idx], (cowW, cowH))

        #if printDebugInfoToScreen: print(f"icon text loc[idx] = {iconCowTextLoc[idx]}")
        font = cv2.FONT_HERSHEY_SIMPLEX; fontScale = 1; fontColor = (255, 0, 255); thickness = 2; lineType = 2
        cv2.putText(outFrame, f"GT{predId}", iconCowTextShadowLoc[idx], font, fontScale, (255,255,255), thickness, lineType)
        cv2.putText(outFrame, f"GT{predId}", iconCowTextLoc[idx], font, fontScale, fontColor, thickness, lineType)

        cv2.putText(outFrame, f"Pr{predId}", predCowTextShadowLoc[idx], font, fontScale, (255, 255, 255), thickness, lineType)
        cv2.putText(outFrame, f"Pr{predId}", predCowTextLoc[idx], font, fontScale, fontColor, thickness, lineType)

    font = cv2.FONT_HERSHEY_SIMPLEX; fontScale = 2; fontColor = (255, 0, 255); thickness = 3; lineType = 2
    #cv2.putText(outFrame, f"Cow Recognition Evaluation", (int(0.25*inFrameOutW), outFrameH-25), font, fontScale, (255, 255, 255), thickness, lineType)
    cv2.putText(outFrame, f"AutoCattleID Inference", (int(0.3*inFrameOutW), outFrameH-25), font, fontScale, (255, 255, 255), thickness, lineType)


    if frameNumber is not None:
        cv2.putText(outFrame, f"Frame no: {frameNumber}", (2, int((outFrameH-inFrameOutH)//2*0.8)), font, fontScale, (255, 255, 255), thickness, lineType)

    #cv2.imshow('outFrame', cv2.resize(outFrame, (640,360)))
    #cv2.imshow('outFrame', cv2.resize(outFrame, (1280,720)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return outFrame


def get_cow_autocattlog_frame(inFrame=[],cropList=[], gtLabelsList =[],  maxCows=4, cowAR = 0.5, frameNumber=None, printDebugInfoToScreen=False):
    """
    This is to get an output frame for saving autocattlog videos, for obsrving how it works.

    Returns a frame that displays
    :param frame: input frame
    :param cropList: list of all cropped cow images

    :param maxCows: maximum number of cows that should be displayed on the side.

    :param frameNumber: to display the frame number from the original video - use later for logging and cross checking

    Note: This code auto scales the input frame based on the max number of cows that are displayed on the side.

    :return: output frame
    """

    outFrameH, outFrameW, outFrameCh = 1080, 1920, 3
    outFrame = np.zeros((outFrameH, outFrameW, outFrameCh), np.uint8)

    predCowLoc = []; iconCowLoc = []
    predCowTextLoc = []; iconCowTextLoc = []
    predCowTextShadowLoc = []; iconCowTextShadowLoc = [] #shadow helps improve readability

    cowH = outFrameH//maxCows
    cowW = int(cowH*cowAR)
    if printDebugInfoToScreen: print(f"outFrame shape = {outFrame.shape}")

    #inframe aspect ratio is assumed to be 16:9
    inFrameOutW = outFrameW - 2*cowW
    inFrameOutW = 16 * int(inFrameOutW/16) #extra space will be used as border on the right
    inFrameOutH = int(9/16 * inFrameOutW)

    if printDebugInfoToScreen: print(f"inframeOutH = {inFrameOutH}, inframeOutW = {inFrameOutW}")

    inFrameLoc = outFrame[(outFrameH-inFrameOutH)//2:(outFrameH-inFrameOutH)//2 + inFrameOutH, :inFrameOutW, :]

    inFrameLoc[:,:,:] = cv2.resize(inFrame, (inFrameOutW, inFrameOutH))

    for i in range(maxCows):
        if printDebugInfoToScreen: print(f"pred locs = {int(i*cowH)}:{int((i+1)*cowH)}, {int(outFrameW-2*cowW)}:{int(outFrameW-1*cowW)}")
        predCowLoc.append(outFrame[int(i*cowH):int((i+1)*cowH), int(outFrameW-2*cowW):int(outFrameW-1*cowW), : ])
        #iconCowLoc.append(outFrame[int(i*cowH):int((i+1)*cowH), int(outFrameW-cowW):, :])

        predCowTextLoc.append((int(outFrameW - 2 * cowW), int((i + 1) * cowH - 1)))
        #iconCowTextLoc.append((int(outFrameW - cowW), int((i + 1) * cowH - 1)))

        predCowTextShadowLoc.append((int(outFrameW - 2 * cowW + 1), int((i + 1) * cowH - 1 + 1)))
        #iconCowTextShadowLoc.append((int(outFrameW - cowW + 1), int((i + 1) * cowH - 1 + 1)))

    for idx in range(min(len(cropList), maxCows)): #gt_labels list is always of non zero len. We get non zero len cropList only if we have a perfect cow detection (all keypoint rule passed after).
        #predId = predIdList[idx]
        #iconImg = cv2.imread(f"{cattlogDirPath}/{predId}.jpg")
        #iconCowLoc[idx][:,:,:] = cv2.resize(iconImg, (cowW,cowH))

        predCowLoc[idx][:, :, :] = cv2.resize(cropList[idx], (cowW, cowH)) # we display what is detected on the frame

        #try:
        gt_label = gtLabelsList[idx]
        #except:
        #    pdb.set_trace()

        #if printDebugInfoToScreen: print(f"icon text loc[idx] = {iconCowTextLoc[idx]}")
        font = cv2.FONT_HERSHEY_SIMPLEX; fontScale = 1; fontColor = (255, 0, 255); thickness = 2; lineType = 2
        #cv2.putText(outFrame, f"GT{predId}", iconCowTextShadowLoc[idx], font, fontScale, (255,255,255), thickness, lineType)
        #cv2.putText(outFrame, f"GT{predId}", iconCowTextLoc[idx], font, fontScale, fontColor, thickness, lineType)

        cv2.putText(outFrame, f"{gt_label}", predCowTextShadowLoc[idx], font, fontScale, (255, 255, 255), thickness, lineType)
        cv2.putText(outFrame, f"{gt_label}", predCowTextLoc[idx], font, fontScale, fontColor, thickness, lineType)

    font = cv2.FONT_HERSHEY_SIMPLEX; fontScale = 2; fontColor = (255, 0, 255); thickness = 3; lineType = 2
    cv2.putText(outFrame, f"AutoCattlogger Output", (int(0.24*inFrameOutW), outFrameH-25), font, fontScale, (255, 255, 255), thickness, lineType)


    if frameNumber is not None:
        cv2.putText(outFrame, f"Frame no: {frameNumber}", (2, int((outFrameH-inFrameOutH)//2*0.8)), font, fontScale, (255, 255, 255), thickness, lineType)

    #cv2.imshow('outFrame', cv2.resize(outFrame, (640,360)))
    #cv2.imshow('outFrame', cv2.resize(outFrame, (1280,720)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return outFrame



if __name__ == "__main__":
    inFrame = cv2.imread("./sampleImg.jpg")
    print(f"inFrame shape = {inFrame.shape}")
    get_cow_matches_frame(inFrame=inFrame, predIdList=['5802','5793','5789','5784'], cattlogDirPath="../../Outputs/top_view_samples/", maxCows=4)

