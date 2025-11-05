'''
Author: Manu Ramesh
This code was initially - ../Experiments/experiment_hierarchical_mapper.py
Since this works comletely and is no longer in the experimental stage, it has been promoted to this folder.

Code for:
-   pixlation, binarization,
-   creating cattlog images & bit vectors from these two operations
-   infering a queried image using the bit vector cattlog
'''





#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html

import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
import numpy as np, pdb, os, glob, pickle, sys

from skimage.filters.rank import entropy
from skimage.morphology import disk, square
#from scipy.stats import entropy
from scipy import stats
from scipy.spatial import distance

import itertools
from collections import Counter
import logging, inspect

# For computing warping cost
# sys.path.insert(0, os.path.abspath('../../dependencies/DynamicSpaceWarping/')) #path to graph warper, clone it to ./ later and change this path accordingly
# from graphWarp import GraphWarper

#can look at other stat values
def getEntropyImage(gray):

    #entropyImg = entropy(gray, disk(3)) #entropy(image, disk(5))
    entropyImg = entropy(gray, square(64))
    return entropyImg

def gridify(gray, pitch = 64, display=True):

    h, w = gray.shape
    outImg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) #gray.copy()

    for i in range(0,h,pitch):
        for j in range(0,w,pitch):
            region = gray[i:i+pitch,j:j+pitch]
            cv2.rectangle(outImg, (j,i), (j+pitch-1, i+pitch-1), color=(255,255,0), thickness=1)

            #val = sum(stats.entropy(region))
            val = np.var(region)
            print(f"variance value = {val}")
            #pdb.set_trace()

            # Using cv2.putText() method
            outImg = cv2.putText(outImg, f'{int(val) if not np.isnan(val) else val}', (j,i+20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,255), thickness=1, lineType=cv2.LINE_AA)

    #cv2.imshow('gray', gray) #original image unaltered
    cv2.imshow('outImg', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pixelate(gray, blkSize=8, type='avg', threshVal=127):

    #you can also do downsize and upsize technique
    #this is smoothing the image. I don't want that
    #h, w = gray.shape
    #outImg = cv2.resize(gray, (w//blkSize, h//blkSize), cv2.INTER_LINEAR)
    #outImg = cv2.resize(outImg, (w,h), cv2.INTER_NEAREST)
    #
    #cv2.imshow(f'pixelated image blk size = {blkSize}', outImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #
    #return outImg

    h, w = gray.shape
    outImg = np.zeros_like(gray)

    fVec = [] #binary feature vector

    for i in range(0,h,blkSize):
        for j in range(0,w,blkSize):
            region = gray[i:i+blkSize,j:j+blkSize]
            outRegion = outImg[i:i+blkSize,j:j+blkSize]

            val = 0
            if type == 'avg':
                val = int(np.average(region))
                #print(f"average = {val}")

            elif type == 'avgThresh':
                #average and threshold
                val = int(np.average(region))
                if val > threshVal:
                    val = 255
                    fVec.append(1)
                else:
                    val = 0
                    fVec.append(0)

            #outRegion = val
            outImg[i:i + blkSize, j:j + blkSize] = val

    #print(f"Feature vector = {fVec}")
    #print(f"len of fVec = {len(fVec)}")

    #cv2.imwrite(f'pixImg_blkSize{blkSize}.png', outImg)
    #cv2.imshow(f'pixelated image blk size = {blkSize}', outImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return outImg, np.array(fVec)


def createTrainingSet(rootDir):
    #creates a training set of pixelated cow values

    strVecDict = {}

    dirs = os.listdir(rootDir)
    for dir in dirs:
        cowID = dir

        imgPaths = glob.glob(f"{rootDir}/{dir}/*.jpg")

        for imgPath in imgPaths:
            gray = cv2.imread(imgPath, 0)
            gray = cv2.resize(gray, (512, 1024))

            blkSizes = [128, 64, 32, 16]

            for blkSize in blkSizes:
                pixImg, fVec = pixelate(gray, blkSize=blkSize, type='avgThresh') #stop at 16
                strVec = ''.join(str(x) for x in fVec)

                if cowID not in strVecDict:
                    strVecDict[cowID] = {f"blk{blkSize}": strVec}
                else:
                    strVecDict[cowID][f"blk{blkSize}"] = strVec


    print(f"strVecDict = {strVecDict}")
    pickle.dump(strVecDict, open('cowDataMatDict.p','wb'))

def createTrainingSet_singleDir(rootDir, saveDir='./', cattlogNameSuffix = "", imgSaveDir=None, colorCorrectionOn=False, threshVal_CC=50):
    #creates a training set of pixelated cow values
    #does everything that createTrainSet() does but works directly on the images in the rootDir rather than in sub dirs
    #i would be using this function to create the bit vectors for the kp-template-aligned cow images
    '''

    :param rootDir: dir that has the images that need to be processed
    :param saveDir: dir to which the bit vector cattlog dictionary is saved
    :param cattlogNameSuffix: suffix given to name of bit vector cattlog
    :param imgSaveDir: Directory in which to save the pixelated images - you could use this later to evaluate predictions

    #for project cow lighting
    :param colorCorrectionOn: set this to True when working with color corrected images. (For project CowLighting beyond SURABHI)
    :threshVal_CC: The thresholding value to be used for binarizing color corrected images.

    :return:
    '''


    if imgSaveDir is not None:
        os.makedirs(imgSaveDir, exist_ok=False) #Exists ok is False to prevent overwriting. This data could be important.

    #threshold for binarization 
    threshVal = 127  # the default, until SURABHI
    if colorCorrectionOn:
        threshVal = threshVal_CC #50


    strVecDict = {}

    imgPaths = glob.glob(f'{rootDir}/*.jpg')
    print(f"Discovered cow images = {imgPaths}")

    for imgPath in imgPaths:

        cowID = imgPath.split('/')[-1].split('.')[0]
        print(f"Currently processsing cow {cowID}")

        gray = cv2.imread(imgPath, 0)
        gray = cv2.resize(gray, (512, 1024))

        blkSizes = [128, 64, 32, 16]

        for blkSize in blkSizes:
            pixImg, fVec = pixelate(gray, blkSize=blkSize, type='avgThresh', threshVal=threshVal) #stop at 16

            #save pixelated images if asked to
            if imgSaveDir is not None:
                #cv2.imwrite(f"{imgSaveDir}/{cowID}_blk{blkSize}.jpg", pixImg) #if there are multiple images of same cow, saves only last one (i am not using this function where there are multiple images of the same cow though)
                cv2.imwrite(f"{imgSaveDir}/{cowID}_blk{blkSize}.png", pixImg) #if there are multiple images of same cow, saves only last one (i am not using this function where there are multiple images of the same cow though) - saving higher quality image

            strVec = ''.join(str(x) for x in fVec)

            if cowID not in strVecDict:
                strVecDict[cowID] = {f"blk{blkSize}": strVec}
            else:
                strVecDict[cowID][f"blk{blkSize}"] = strVec


    print(f"strVecDict = {strVecDict}")
    pickle.dump(strVecDict, open(f'{saveDir}/cowDataMatDict_KP_aligned{cattlogNameSuffix}.p','wb'))


def inferImage(strVecDict, gray, K=3, gt_cowID=None, printDebugInfoToScreen=False, threshVal=127, queryBitVecsDict={}, topKHamDistDict={}):
    '''
    Compares features generated from input image (and its transformed versions) to all cow features in dictionary.
    Does this at all block sizes. Fetches top K predictions at each block size.
    Gets the cowID that appears max number of times in topK across all  block sizes and transformed query images)
    Confidence is proportional to number of block sizes in which the cow has been matched.

    :param strVecDict: dictionary of string feture vectors (binary runs)
    :param gray: grayscale image
    :param K: K in top K values
    :param gt_cowID: ground truth cowID - give this arg if you want to calculate the top k accuracy, - if not None, returns the index of the gt in the sorted predictions.
    :param threshVal: The value to threshold pixel block as black or white. Use 127 for results till SURABHI. For cow lighting experiments, use values around 50 (still working on it).
    :param queryBitVecsDict: Pass any dictionary by reference if you need the bit vectors output.
    :param topKHamDistDict: Pass any dict by reference if you need the hamming distances between the queried cow instance and the top K predicted cow instances.
    :return: predicted cowID and confidence, gt_locs -> location of ground truth in prediction list - used to compute top K accuracy
    '''

    #pass the dict, not path to the dict
    #also check inverted cows
    #Not implemented yet: accept masks and weigh the matchings accordingly

    if len(gray.shape) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (512, 1024))

    #gray180 = cv2.rotate(gray, cv2.ROTATE_180)

    #queryList = [gray, gray180]
    queryList = [gray] #180 not needed if kp aligned

    #blkSizes = [128, 64, 32, 16]
    #blkSizes = [16, 32] #reducing the block sizes assuming cows are near perfectly algined after KP template matching, order matters - go from finest level detail to coarser - smaller blk sizes must come first
    blkSizes = [16]
    topK_cowIDs = []
    gt_locs = [-1]*len(blkSizes) #used to store location of ground truth values - used to calculate top K accuracies for all K all at once!

    for idx, blkSize in enumerate(blkSizes):

        for queryGray in queryList:
            pixImg, query_fVec = pixelate(queryGray, blkSize=blkSize, type='avgThresh', threshVal=threshVal)  # stop at 16
            hamDistance_list = []

            queryBitVecsDict[blkSize] = query_fVec

            for cowID in strVecDict:
                strVec = strVecDict[cowID][f"blk{blkSize}"]
                key_fVec = [int(x) for x in strVec]
                hamDistance = distance.hamming(key_fVec, query_fVec)
                hamDistance_list.append(hamDistance)

        #sortedCowIDs = [x for x, _ in sorted(zip(list(strVecDict.keys())*len(queryList), hamDistance_list), key= lambda y: y[1])] #ascending order
        sortedCows = [(x,y) for x, y in sorted(zip(list(strVecDict.keys())*len(queryList), hamDistance_list), key= lambda z: z[1])] #ascending order
        sortedCowIDs = [x for x, y in sortedCows]
        sortedHamDists = [y for x, y in sortedCows]

        #if printDebugInfoToScreen: print(f"sorted cow IDS = {sortedCowIDs}")

        if gt_cowID is not None:
            try:
                gt_locs[idx] = sortedCowIDs.index(gt_cowID)
            except:
                #gt_locs[idx] = -1 #= -1 by default
                pass

        top1_cowID = sortedCowIDs[0]

        topK_cowIDs.append(sortedCowIDs[:K])

        topKHamDistDict[f"blk{blkSize}"] = sortedHamDists[:K]

        #pdb.set_trace()

        if printDebugInfoToScreen: print(f"BLK{blkSize}: Top1 cow = {top1_cowID}, hamming distance = {min(hamDistance_list)}")
        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> BLK{blkSize}: Top1 cow = {top1_cowID}, hamming distance cows with query image = {min(hamDistance_list)}")

    #topK_cowIDs = np.array(topK_cowIDs)
    if printDebugInfoToScreen: print(f"top {K} cowIDs array = {topK_cowIDs}, Hamming Distances of top {K} cows = {topKHamDistDict}")
    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> top {K} cowIDs array = {topK_cowIDs}, Hamming Distances of top {K} cows with query image = {topKHamDistDict}")

    topK_merged = list(itertools.chain(*topK_cowIDs)) #converts list of lists to a single list
    counts = Counter(topK_merged) #counter should also sort items in decreasing order of counts
    counts = list(counts.items())

    top1_cowID, top1_confidence = counts[0]
    top1_confidence = top1_confidence / len(blkSizes) * 100
    #pdb.set_trace()

    if printDebugInfoToScreen: print(f"The final top 1 cowID = {top1_cowID}, confidence = {top1_confidence}")
    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> The final top 1 cowID = {top1_cowID}, confidence = {top1_confidence}")

    return top1_cowID, top1_confidence, gt_locs


def inferBitVector(strVecDict, queryBitVector, K=3, gt_cowID=None, printDebugInfoToScreen=False, topKHamDistDict={}, topKWarpDistDict={}, sortTopK_byWarpingCost=False, gw_ROI_Mask=None):
    '''
    Same as inferImage() but works with bit vectors directly. This is useful when you have the bit vectors already and don't want to pixelate the image again.

    :param strVecDict: dictionary of string feature vectors (binary runs) = the cattlog
    :param queryBitVector: (str) The bitvector that needs to be queried. Should be that of blkSize 16.
    :param K: K in top K values
    :param gt_cowID: ground truth cowID - give this arg if you want to calculate the top k accuracy, - if not None, returns the index of the gt in the sorted predictions.
    :param topKHamDistDict: Pass any dict by reference if you need the hamming distances between the queried cow instance and the top K predicted cow instances.
    :param topKWarpDistDict: Pass any dict by reference if you need the warping distances between the queried cow instance and the top K predicted cow instances.
    :param sortTopK_byWarpingCost: If True, the top K cows obtained by hamming distance are reordered based on their warping costs with the query bit vector.
    :param gw_ROI_Mask: The ROI mask to be used during graph warping. If None, no mask is used.
    :return: predicted cowID and confidence, gt_locs -> location of ground truth in prediction list - used to compute top K accuracy
    '''

    blkSizes = [16]
    topK_cowIDs = []
    gt_locs = [-1]*len(blkSizes) #used to store location of ground truth values - used to calculate top K accuracies for all K all at once!

    for idx, blkSize in enumerate(blkSizes):


        query_fVec = [int(x) for x in queryBitVector]
        hamDistance_list = []

        for cowID in strVecDict:
            strVec = strVecDict[cowID][f"blk{blkSize}"]
            key_fVec = [int(x) for x in strVec]
            hamDistance = distance.hamming(key_fVec, query_fVec)
            hamDistance_list.append(hamDistance)

        sortedCows = [(x,y) for x, y in sorted(zip(list(strVecDict.keys()), hamDistance_list), key= lambda z: z[1])] #ascending order
        sortedCowIDs = [x for x, y in sortedCows]
        sortedHamDists = [y for x, y in sortedCows]

        sortedCows_byWarp = [] #only top K
        originalOrder_topK = [] #to store original order of top K cows before reordering by warping cost

        #if printDebugInfoToScreen: print(f"sorted cow IDS = {sortedCowIDs}")


        # FUTURE FUNCTIONALITY: Reorder top K cows based on warping cost. COMMENTING FOR NOW.
        # if sortTopK_byWarpingCost:
        #     # Resort the top K cows by warping cost
        #     gw = GraphWarper()
        #     warpingCosts = []
        #     topK_hammingPreds = sortedCowIDs[:K]
        #     for cowID in topK_hammingPreds:
                
        #         #get barcode sizes from len of feature vector, (aspect ratio is always 1:2, w:h)
        #         barcode_W = int(np.sqrt(len(query_fVec) // 2))
        #         barcode_H = 2*barcode_W

        #         queryBarcode = np.array(query_fVec).reshape((barcode_H, barcode_W))  
        #         key_fVec = [int(x) for x in strVecDict[cowID][f"blk{blkSize}"]]
        #         keyBarcode = np.array(key_fVec).reshape((barcode_H, barcode_W))

        #         # Kernel size defines the search neighborhood for finding correspondences, larger values allow for more flexibility but increase computation time.
        #         _, warpingCost = gw.apply_warp(queryBarcode, keyBarcode, kernel_size=15, display_graph=False, improve_match_spread=True, penalize_farther_matches=True, max_absorption_per_pixel=None, ROI_Mask=gw_ROI_Mask, draw_connections=False)
        #         # _, warpingCost = gw.apply_warp(queryBarcode, keyBarcode, kernel_size=3, display_graph=False, improve_match_spread=True, penalize_farther_matches=True, max_absorption_per_pixel=None, ROI_Mask=gw_ROI_Mask, draw_connections=False)
                
        #         warpingCosts.append(warpingCost)
        #     sortedCows_byWarp = [(x,y) for x, y in sorted(zip(topK_hammingPreds, warpingCosts), key= lambda z: z[1])] #ascending order
            
        #     reorderedCowIDs = [x for x, y in sortedCows_byWarp]
        #     originalOrder_topK = sortedCowIDs[:K]

        #     if reorderedCowIDs != originalOrder_topK:
        #         if printDebugInfoToScreen:
        #             print(f"TopK Prediction order changed based on graph warping distance!")
        #         logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> TopK Prediction order changed based on graph warping distance! Original order (by Hamming) = {sortedCowIDs[:K]}, New order (by Warping) = {reorderedCowIDs}")

        #         if reorderedCowIDs[0] != originalOrder_topK[0]:
        #             if printDebugInfoToScreen:
        #                 print(f"Top1 Prediction changed based on graph warping distance!")
        #             logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Top1 Prediction changed based on graph warping distance! Original Top1 (by Hamming) = {originalOrder_topK[0]}, New Top1 (by Warping) = {reorderedCowIDs[0]}")

        #     sortedCowIDs = reorderedCowIDs + sortedCowIDs[K:]
            
        #     topKWarpingDists = [y for x, y in sortedCows_byWarp]

        #     # sortedHamDists = [y for x, y in sortedCows_byWarp] + sortedHamDists[K:] # Do not mix warping costs with hamming distances

        #     logging.debug(f"Sorted Ham Dists before reordering by warping cost: {sortedHamDists[:K]}")

        #     sortedHamDists = [x for x, y in sorted(zip(sortedHamDists[:K], warpingCosts), key=lambda z: z[1])] + sortedHamDists[K:]  # Reorder top K hamming distances

        #     logging.debug(f"Sorted Ham Dists after reordering by warping cost: {sortedHamDists[:K]}")



        if gt_cowID is not None:
            try:
                gt_locs[idx] = sortedCowIDs.index(gt_cowID)
            except:
                #gt_locs[idx] = -1 #= -1 by default
                pass

        top1_cowID = sortedCowIDs[0]

        topK_cowIDs.append(sortedCowIDs[:K])

        topKHamDistDict[f"blk{blkSize}"] = sortedHamDists[:K]
        # topKWarpDistDict[f"blk{blkSize}"] = topKWarpingDists[:K]

        #pdb.set_trace()

        if printDebugInfoToScreen: print(f"BLK{blkSize}: Top1 cow = {top1_cowID}, hamming distance = {sortedHamDists[0]}")
        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> BLK{blkSize}: Top1 cow = {top1_cowID}, hamming distance cows with query image = {sortedHamDists[0]}")
        
        # Repeat
        # if sortTopK_byWarpingCost and printDebugInfoToScreen:
        #     print(f"BLK{blkSize}: Top1 cow after reordering by warping cost = {sortedCowIDs[0]}, warping distance = {min(topKWarpingDists)}")
        # logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> BLK{blkSize}: Top1 cow after reordering by warping cost = {sortedCowIDs[0]}, warping distance cows with query image = {min(topKWarpingDists)}")

    #topK_cowIDs = np.array(topK_cowIDs)
    if printDebugInfoToScreen: print(f"Top {K} cowIDs array = {topK_cowIDs}, Hamming Distances of top {K} cows = {topKHamDistDict}")
    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Top {K} cowIDs array = {topK_cowIDs}, Hamming Distances of top {K} cows with query image = {topKHamDistDict}")

    if sortTopK_byWarpingCost:
        if printDebugInfoToScreen: print(f"Top {K} cowIDs array after reordering by warping cost = {topK_cowIDs}, Warping Distances of top {K} cows = {topKWarpDistDict}")
        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Top {K} cowIDs array after reordering by warping cost = {topK_cowIDs}, Warping Distances of top {K} cows with query image = {topKWarpDistDict}")
        #If the cows have been reordered based on warping cost, the topK hamming distances will not be in ascending order anymore.

    topK_merged = list(itertools.chain(*topK_cowIDs)) #converts list of lists to a single list
    counts = Counter(topK_merged) #counter should also sort items in decreasing order of counts
    counts = list(counts.items())

    top1_cowID, top1_confidence = counts[0]
    top1_confidence = top1_confidence / len(blkSizes) * 100
    #pdb.set_trace()

    if printDebugInfoToScreen: print(f"The final top 1 cowID = {top1_cowID}, confidence = {top1_confidence}")
    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> The final top 1 cowID = {top1_cowID}, confidence = {top1_confidence}")

    return top1_cowID, top1_confidence, gt_locs


def findSameBitVecCows(bitVecCattlog=None, bitVecCattlogPath=None):
    '''
    Function to find cows with sharing the same bitvectors. This usually happens if the cows are of uniform coloration - such as completely black or completely white.
    This is designed to work with block size of 16 X 16 (in an image of size 512X1024 that is scaled up from 256X512 - as to why this is scaled - it was an error made a long time ago in EI2023 paper - don't worry. It still works fine.).
    :param bitVecCattlogPath: path to bit vector cattlog.
    :param bitVecCattlog: the bit vector cattlog dictionary.

    bitVecCattlog is prefered. If bitVecCattlog is None, then the fn attempts to load from file at bitVecCattlogPath.

    :return: sameBVecCows - a list, sameBVecCowsDict - a dictionary
    '''

    assert (bitVecCattlog is not None) or (bitVecCattlogPath is not None), 'User must pass either one of the two input values'

    if bitVecCattlog is not None:
        cattlogDict = bitVecCattlog
    else:
        cattlogDict = pickle.load(open(bitVecCattlogPath, 'rb'))

    invDict = {} #inverse mapping. bitVec->cowID

    for k, v in cattlogDict.items():
        invDict[v['blk16']] = invDict.get(v['blk16'], []) + [k]

    #now use this inv dictionary to create a dict cowID->[other cows with same bitVec]
    sameBVecCowsDict = {}
    sameBVecCows = [] #just to all cows sharing same bVecs together

    for k, v in invDict.items():
        if len(v) >1: #if there is more than one cow for a bVec

            sameBVecCows.append(v)

            for cowID in v:
                v2 = v.copy()
                v2.remove(cowID)
                sameBVecCowsDict[cowID] = v2
                #pdb.set_trace()

    return  sameBVecCows, sameBVecCowsDict


if __name__ == "__main__":
    #img = cv2.imread("./images/5982.jpg")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (448,896))
    #gray = cv2.resize(gray, (512, 1024))

    #ret, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    #cv2.imshow('gray', gray)
    #cv2.imshow('thresh1', thresh1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #'''
    #pixImg, fVec = pixelate(gray, blkSize=16, type='avgThresh')
#
    #print(f"h Distance = {distance.hamming(fVec,fVec)}")
    #pdb.set_trace()

    #fVec2 = fVec.copy()
    #fVec2[3:50] = 0
    #print(f"h Distance = {distance.hamming(fVec,fVec2)}")
    #'''

    #ret, thresh2 = cv2.threshold(pixImg, 128, 255, cv2.THRESH_BINARY)

    #gridify(gray, pitch=128)
    #gridify(gray, pitch=64)

    ####################################################################################################################

    #TO CREATE TRAINING SET
    #createTrainingSet(rootDir='../../Outputs/Cow_TV_Crops_Frozen_Whole_noEmptyDirs/')
    #createTrainingSet_singleDir(rootDir=f"../../Outputs/top_view_KP_aligned_cattlog_june_08_2022/", imgSaveDir=f"../../Outputs/top_view_KP_aligned_pixelatedImgs_june_08_2022/") #training set
    #createTrainingSet_singleDir(rootDir=f"../../Outputs/top_view_KP_aligned_cattlog_june_08_2022_CC/", imgSaveDir=f"../../Outputs/top_view_KP_aligned_pixelatedImgs_june_08_2022_CC/", colorCorrectionOn=True) #training set - Color Corrected (beyond SURABHI)

    #Testing on test set bit vectors
    #createTrainingSet_singleDir(rootDir=f"../../Outputs/top_view_KP_aligned_cattlog_june_09_2022_v4_test/", imgSaveDir=f"../../Outputs/top_view_KP_aligned_pixelatedImgs_june_09_2022/") #test set - for end to end evaluation

    #Inferring From an image
    #strVecDict = pickle.load(open('./cowDataMatDict.p', 'rb'))
    #strVecDict = pickle.load(open('./cowDataMatDict_KP_aligned_june_08_2022.p', 'rb'))
    #inferImage(strVecDict, gray, K = 5)

    ####################################################################################################################
    #finding cows with common bit vectors
    #sameBVecCows, sameBVecCowsDict = findSameBitVecCows(bitVecCattlogPath='./Models/bit_vector_cattlogs/CowLightingBitVectors/cowDataMatDict_KP_aligned_june_08_2022_CC-thresh50.p')
    sameBVecCows, sameBVecCowsDict = findSameBitVecCows()
    print(f"Cows with same bitVectors = \n{sameBVecCows}\n\nDict of cows with shared bitVectors = \n{sameBVecCowsDict}")

