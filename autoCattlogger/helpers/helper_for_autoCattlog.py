'''
Author: Manu Ramesh

Has helper functions for autoCatalogging.

'''

import pickle, numpy as np
import logging
from shapely.geometry import Polygon #for finding bbox centroids
from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors
import pdb, pandas as pd
import os, cv2


#############################################################################################################
################ POST PROCESSING FUNCTIONS TO FILTER OUT TRACKS AND COMPUTE AUTO CATTLOG ####################

def postProcessTracks1(tracksList, frameW=1920, printDebugInfoToScreen=False, requiredDirection="LR"):
    '''
    This is a sample post processing function to filter out tracks that start from the left and end on the right.
    You can write your custom post processing function based on your requirements using this function as a reference.    

    :param tracksList: list of tracks which includes track points  (open, closed, or required tracks) - in the same format as that of the output of the cattlogFromVideosList_multiInstances (ACv2) function in autoCattlog.py.
    :param frameW: width of the frame (default is 1920) - used to find the entry and exit directions of the cow.
    :param printDebugInfoToScreen: boolean, if True, prints the debug info to the screen. (default is False)
    :param requiredDirection: direction to filter the tracks (default is "LR" Left to right) - keep cows that enter from the left and exit on the right.

    :return: requiredTracks: list of tracks that start from the left and end on the right (or center) - in the same format as that of the output of the cattlogFromVideosList_multiInstances (ACv2) function in autoCattlog.py.
    :return: bitVecCattlogDict: dictionary of bit vectors (serialized cow barcodes) for the required tracks.
    '''

    logging.info(f"Post processing tracks. FrameW = {frameW}.")
    if frameW is None:
        frameW = 1920
        logging.info(f"Frame width not provided. Using default value of 1920.")

    logging.info(f"***************************************")
    logging.info(f"Post Processing Tracks\n")
    logging.info(f"There are {len(tracksList)} tracks in total.")
    if printDebugInfoToScreen: print(f"There are {len(tracksList)} tracks in total.")

    tracksWithCows = [x for x in tracksList if x['hasCow']]
    logging.info(f"There are {len(tracksWithCows)} tracks with cows.")
    if printDebugInfoToScreen: print(f"There are {len(tracksWithCows)} tracks with cows.")

    #compute entry and exit directions of cows
    def getBboxLocation(boxPts,frameW=1920):
            '''
            Takes in bbox corner (x,y) coordinate pairs and returns 'left' or 'right' if the the box is closer to left or right edge of the frame respectively.
            Can use this info to filter cows that enter from the right side (in the reverse direction).
            Can add more complexity to this check later.
            :param boxPts: list of 4 corner points of the bbox (x,y)
            :param frameW: width of the frame
            :return: 'left' or 'right'
            '''
            p = Polygon(boxPts)
            x,y = p.centroid.coords[0]

            if x < frameW/3:
                return 'left'
            elif x < 2*frameW/3:
                #this case is encountered when the video starts or ends abruptly and the cow is in the center of the frame
                #This happened at the end of the second hour long video in Sam 2023 Lameness data's July Videos.
                #As of now, we only check if the cow has ended in the center of the frame. 
                #You can later modify the condition below to look for the start of the cow in the center of the frame.
                return 'center' 
            else:
                return 'right'
            
    for track in tracksWithCows:
        #get entry and exit directions - currently left/right/center
        track['entryDirection'] = getBboxLocation(track['trackPoints'][0]['rotatedBBOX_locs'],frameW)
        track['exitDirection']  = getBboxLocation(track['trackPoints'][-1]['rotatedBBOX_locs'],frameW)
    
    #pdb.set_trace()

    if requiredDirection == "LR":
        #use only those tracks that start from the left and end at the right
        requiredTracks = [x for x in tracksWithCows if x['entryDirection']=='left' and (x['exitDirection']=='right' or x['exitDirection']=='center')]
        logging.info(f"There are {len(requiredTracks)} tracks that start from the left and end on the right (or center).")
        if printDebugInfoToScreen: print(f"There are {len(requiredTracks)} tracks that start from the left and end on the right.")
    else: #'RL'
        requiredTracks = [x for x in tracksWithCows if x['entryDirection']=='right' and (x['exitDirection']=='left' or x['exitDirection']=='center')]
        logging.info(f"There are {len(requiredTracks)} tracks that start from the right and end on the left (or center).")
        if printDebugInfoToScreen: print(f"There are {len(requiredTracks)} tracks that start from the right and end on the left.")

    #compute autoCattlog bit-vectors for the required tracks

    bitVecCattlogDict = {} #we are only processing blkSize 16 as of now

    for track in requiredTracks:
        #Method 1:
        # Final bit vector = bit-wise statistical mode of all bit vectors. => majority vote for white or black at each pixel block.

        bitVecList = [x['bitVecStr'] for x in track['trackPoints'] if x['bitVecStr'] != '']
        modeBitVec = stats.mode(bitVecList).mode #has .counts with the counts of each entry

        #if printDebugInfoToScreen: print(f"ModeBitVec = {modeBitVec}")

        finalBitVecStr = ''.join(modeBitVec[0].tolist()) #saving bit vector as a string of bits
        
        track['autoCattBitVecStr'] = finalBitVecStr

        bitVecCattlogDict[f"trackId_{track['trackId']}"] = {'blk16':finalBitVecStr}

    #return the required tracks and the bit-vectors dictionary
    #can add another function to associate the trackIDs to the GT_labels


    return requiredTracks, bitVecCattlogDict

#Wrapper post processing fns - in case I change the implementation to accept a post processing fn as an arg, you can pass this
def postProcessTracks_RL(tracksList, frameW=1920, printDebugInfoToScreen=False, requiredDirection="RL"):
    #get the required tracks
    requiredTracks, bitVecCattlogDict = postProcessTracks1(tracksList, frameW=frameW, printDebugInfoToScreen=printDebugInfoToScreen, requiredDirection=requiredDirection)

    return requiredTracks, bitVecCattlogDict


#############################################################################################################
######################## OTHER UTILITY FUNCTIONS FOR AUTO CATTLOG  V2 #######################################

def getCutsInfoList(tracksList, saveDir=None):
    '''
    To generate the CSV file with the start and end frame info for each cow in the track.
    This is useful for generating the cut videos in other synchronized cameras looking at the cows from other views.

    :param tracksList: list of tracks which includes track points  (required tracks) - in the same format as that of the output of the cattlogFromVideosList_multiInstances (ACv2) function in autoCattlog.py.
    :param saveDir: directory to save the CSV file (default is None, which means do not save the file)

    :return: df: pandas dataframe with the start and end frame info for each cow in the track.

    '''

    _tracks = tracksList.copy()

    for track in _tracks:
        if 'sampleCropImg' in track:
            del track['sampleCropImg']
    
        track['startVideo'] = track['trackPoints'][0]['videoName']
        track['endVideo'] = track['trackPoints'][-1]['videoName']
        del track['trackPoints']
    
    df = pd.DataFrame(_tracks)

    if saveDir is not None:
        df.to_csv(f"{saveDir}/autoCattlog_cutsList.csv", index=False)



    return df


def attachGTLabels(gtLabelsCSV_path, tracksList, saveDir='./', sampleCropImgsDir=None, bitVecCattlogPath=None):
    '''
    Attach the GT labels to the tracks, the sample crop images, and the bit-vector cattlog.

    :param gtLabelsCSV_path: path to the GT labels CSV file (should have a column 'CowID' with the GT labels - the first row should be the header with 'CowID' in it)
    :param tracksList: list of tracks which includes track points  (required tracks) - in the same format as that of the output of the cattlogFromVideosList_multiInstances (ACv2) function in autoCattlog.py.
    :param saveDir: directory to save the tracks with GT labels (default is './')
    :param sampleCropImgsDir: directory to save the sample crop images with GT labels (cowID) as the name in jpg format. (default is None, which means do not save the images)
    :param bitVecCattlogPath: path to the bit-vector cattlog dictionary (default is None, which means do not save the dictionary)

    :return: tracksList: list of tracks with GT labels - in the same format as that of the output of the cattlogFromVideosList_multiInstances (ACv2) function in autoCattlog.py.
    :return: bvCattlogWithGTLabels: dictionary of bit vectors (serialized cow barcodes) for the tracks with GT labels (default is None, which means do not process the dictionary)
    '''

    bvCattlogWithGTLabels = None

    #load the GT labels
    #these should be in the order of appearance, which is the same order as tracksList
    gt_labels = pd.read_csv(gtLabelsCSV_path)

    assert len(tracksList) == len(gt_labels), "Number of tracks and GT labels do not match."

    
    for i, track in enumerate(tracksList):
        track['gt_label'] = str(gt_labels['CowID'][i])
    
    _tracks = tracksList.copy()

    if sampleCropImgsDir is not None:
        #rename images to have the gtLabels
        #Let us save the sample crop images even for the Ignore cows. This can help us find cows in the videos.
        
        outImagesDir = f"{os.path.split(sampleCropImgsDir)[0]}/autoCattlogV2_sampleCrops_withGTLabels/"
        #pdb.set_trace()
        os.makedirs(outImagesDir, exist_ok=True)

        logging.info(f"Copying sample crop images to a new directory with GT labels as filenames.\nThese are at: {outImagesDir}")

        for track in _tracks:
            trackId = track['trackId']
            gtLabel = track['gt_label']
            os.system(f"cp -al '{sampleCropImgsDir}/trackId_{trackId}.jpg' '{outImagesDir}/{gtLabel}.jpg'")
            
    
    #Let's get rid of the 'ignore' cows in the tracks list too
    tracksList = [x for x in tracksList if 'IGNORE' not in x['gt_label'].upper()]    
    pickle.dump(tracksList, open(f"{saveDir}/tracks_withGTLabels.pkl", 'wb'))


    if bitVecCattlogPath is not None:
        #replace trackIDs with GTLabels
        cattlogDict = pickle.load(open(bitVecCattlogPath, 'rb'))
        outDict = {}

        for track in _tracks:
            trackId = track['trackId']
            gtLabel = track['gt_label']

            #Ignoring the cows without GT labels
            #such cows have 'IGNORE' in their GT labels
            if 'IGNORE' not in gtLabel.upper():
                outDict[gtLabel] = cattlogDict[f"trackId_{trackId}"]
                    
        outFilePath = f"{os.path.split(bitVecCattlogPath)[0]}/cowDataMatDict_autoCattlogV2_withGTLabels.p"
        logging.info(f"Saving the bit-vector cattlog dictionary with GT labels as keys. This is at: {outFilePath}")   

        pickle.dump(outDict, open(outFilePath, 'wb'))

        bvCattlogWithGTLabels = outDict

        #pdb.set_trace()
    

    return tracksList, bvCattlogWithGTLabels


def createCattlogImages_fromBitVectors(bvCattlog, outDir='./cattlogBarcodeImages/'):
    '''
    Creates cattlog barcode images from bit vector cattlog.
    I am creating this function to create images for autoCat bit vectors - to use them in the paper.

    :param bvCattlog: dictionary of bit vectors (serialized cow barcodes) for the tracks with GT labels (default is None, which means do not process the dictionary)
    :param outDir: directory to save the images (default is None, which means do not save the images)
    '''

    def createImageFromBV(bVec, blkSize, imgH, imgW):
        '''
        For creating image from bit vector.
        :param blkSize: int, block size
        :param imgH: height of the image used to create the bVec, and the image to be returned
        :param imgW: width of the image used to create the bVec, and the image to be returned
        :param outDir: directory to save the outputs

        :return: img- numpy array
        '''

        h = imgH//blkSize
        w = imgW//blkSize

        img = np.array([int(x) for x in bVec])
        img.resize((h,w))
        #pdb.set_trace()
        img = (img*255).astype(np.uint8)
        img = img.repeat(blkSize, axis=0).repeat(blkSize, axis=1)

        return img


    os.makedirs(outDir, exist_ok=True)

    for cowID, bVecsDict in bvCattlog.items():
        
        #print(f"CowID = {cowID}")

        for blkSize, bv in bVecsDict.items():

            blkSizeInt = int(blkSize.split('blk')[-1])
            img = createImageFromBV(bv, blkSize=blkSizeInt, imgH=1024, imgW=512)

            cv2.imwrite(f"{outDir}/{cowID}_{blkSize}.jpg", img)



if __name__ == '__main__':
    #tracksList = pickle.load(open('./autoCattlogMultiCow_outputs/closedTracks1.pkl', 'rb'))
    tracksList = pickle.load(open('./experiments_for_autoCattlogV2/Results for AutoCattlogV2/autoCattlogV2_2023SamData/autoCattlogV2_lameness-23-07/closedTracks.pkl', 'rb'))
    rt, bvd = postProcessTracks1(tracksList, printDebugInfoToScreen=True)
    
    pdb.set_trace()

    #tracksList = pickle.load(open('./experiments_for_autoCattlogV2/autoCattlogV2_S22Day2_CC/requiredTracks.pkl', 'rb'))
    #getCutsInfoList(tracksList, saveDir='./experiments_for_autoCattlogV2/autoCattlogV2_S22Day2_CC/')
    

    #pdb.set_trace()