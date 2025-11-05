'''
Author: Manu Ramesh
Contains functions to help in calculations while inferring things from Keypoints
'''




import numpy as np
import cv2, pdb, os, glob, pickle
import logging, inspect, sys

sys.path.insert(0, '../../') #top level dir
from autoCattlogger.helpers.helper_for_stats import DatasetStats


# if __name__ == "__main__":
#     from helper_for_stats import DatasetStats
# else:
#     from .helper_for_stats import DatasetStats

#later versions create import errors
try:
    from ndicts.ndicts import DataDict, NestedDict #https://github.com/edd313/ndicts for comparing nested dictionaries
except ImportError:
    from ndicts import DataDict, NestedDict  # https://github.com/edd313/ndicts for comparing nested dictionaries

def printAndLog(strVal, logLevel = 'info',  printDebugInfoToScreen=False):
    '''
    Function to print and log. Just to reduce clutter.
    :param strVal: input string value
    :param logLevel: level for logger - info, debug, error, warning ...
    :return: None
    '''
    if printDebugInfoToScreen: print(strVal)
    if logLevel == 'info':
        logging.info(strVal)
    elif logLevel == 'debug':
        logging.debug(strVal)
    elif logLevel == 'error':
        logging.error(strVal)


def log_cow_det_reid_stat(countsDict, kpRulesPassCounts, nKPDetectionCounts):
    '''
    Used for logging cow detection and reid stats

    :param countsDict: Dictionary of the form {'total_seenFrames': 0, 'total_framesWithAtLeaset1bbox':0, 'total_instanceBBoxes':0, 'total_AR_area_pass':0, 'total_allKps_det_inter':0, 'total_allKPRulePass':0, 'total_correct_preds':0, and more}
    :param kpRulesPassCounts: 1D NP array with number of images with index number of keypoint rule passes (a histogram)
    :param nKPDetectionCounts: 1D NP array of size number of keypoints + 1, containing the histogram of number of visible keypoints - these are the keypoints that are predicted to be visible (not ground truth visible)
    :return:
    '''

    logging.info('os.path.basename(__file__)}>{inspect.stack()[0][3]}>> \n\n*****************COW DET AND REID STAT**************\n\n')
    print('\n\n*****************COW DET AND REID STAT**************\n\n')
    logging.info(f'Total Seen Frames = {countsDict["total_seenFrames"]}')
    print(f'Total Seen Frames = {countsDict["total_seenFrames"]}')
    logging.info(f'Total frames with at least one detection = {countsDict["total_framesWithAtLeaset1bbox"]}')
    print(f'Total frames with at least one detection = {countsDict["total_framesWithAtLeaset1bbox"]}\n')

    total_seenFrames = countsDict.pop("total_seenFrames")
    total_framesWithAtLeaset1bbox = countsDict.pop("total_framesWithAtLeaset1bbox")
    gt_locs_list = countsDict.pop('gt_locs_list')
    total_instances = countsDict['total_instanceBBoxes']

    topK_VideoLevelDict = countsDict.pop('topK_VideoLevelDict') if 'topK_VideoLevelDict' in countsDict else None #need to pop this value before the for loop that follows
    nVideos = countsDict.pop('nVideos') if 'nVideos' in countsDict else 0 #non 0 if testing on video dataset
    nImages = countsDict.pop('nImages') if 'nImages' in countsDict else 0 #non 0 if testing on image dataset
    cowsWithNoCorrectPreds_list = countsDict.pop('cowsWithNoCorrectPreds_list') if 'cowsWithNoCorrectPreds_list' in countsDict else None #This is a list of cows with no correct top-1 instance level prediction on any frame of their cut-videos.

    nKpCorrectedInstances = countsDict.pop('total_kpCorrected_instances') if 'total_kpCorrected_instances' in countsDict else 0

    total_disc_correct_preds = countsDict.pop('total_disc_correct_preds') if 'total_disc_correct_preds' in countsDict else 0 #discounted correct predictions

    _ = countsDict.pop('frameLevelEvalInfoList') if 'frameLevelEvalInfoList' in countsDict else None #not needed for logging. Must remove before moving on.


    keysString = "Keys:\t\t\t\t|\t"
    valsString = "Vals:\t\t\t\t|\t\t"
    percentOfPreviousValString =       "Percent of previous value:\t|\t\t"
    percentOfTotalInstancesString = "Percent of total instances:\t|\t\t"

    previousVal = countsDict['total_instanceBBoxes']

    print(f"counts dict = {countsDict}")

    for k, v in countsDict.items():
        keysString += f"{k}\t|\t"
        valsString += f"{v}\t\t|\t\t"

        percentOfPrevVal = v/previousVal * 100.0 if previousVal != 0 else 0
        percentOfPreviousValString += f"{percentOfPrevVal:0.2f}\t\t|\t\t"

        percentOfTotalInstances = v/total_instances * 100.0 if total_instances != 0 else 0
        percentOfTotalInstancesString += f"{percentOfTotalInstances:0.2f}\t\t|\t\t"

        previousVal = v


    hLineString = "-" * len(keysString.replace("\t","    "))

    outStrList = ["\n", hLineString, keysString, valsString, percentOfPreviousValString, percentOfTotalInstancesString, hLineString]
    outString = '\n'.join(outStrList)
    logging.info(outString)
    print(outString)

    #print number of kp corrected instances
    printAndLog(f"Number of kp corrected instances = {nKpCorrectedInstances}. This is {nKpCorrectedInstances/(countsDict['total_allKPRulePass']+1e-10)*100:0.2f}% of instances passing all rules.", printDebugInfoToScreen=True)
    printAndLog(f"Note: An instance is declared KP corrected only if it passes all rules upon correction.\n", printDebugInfoToScreen=True)

    #print the number of discounted predictions
    printAndLog(f"\nNumber of discounted instances = {total_disc_correct_preds}.", printDebugInfoToScreen=True)
    printAndLog(f"Note: A discounted instance is one where the system predicted an ID of another which has the same bit vector as that of the ground truth cow.\n", printDebugInfoToScreen=True)


    #now for the kp rule pass counts histogram
    tabSpace = "\t"
    logging.info("KP Rule pass counts histogram"); print("KP Rule pass counts histogram")
    logging.info(f"Number of rules  : {tabSpace.join(str(x) for x in np.arange(len(kpRulesPassCounts)))}"); print(f"Number of rules: {tabSpace.join(str(x) for x in np.arange(len(kpRulesPassCounts)))}")
    logging.info(f"Counts of passes : {tabSpace.join(str(x) for x in kpRulesPassCounts)}"); print(f"Counts of passes : {tabSpace.join(str(x) for x in kpRulesPassCounts)}")

    #now for number of KPs detected per instance histogram
    logging.info("\n\nNumber of keypoints visible (predicted by network) per instance histogram"); print("\n\nNumber of keypoints visible (predicted) per instance histogram")
    logging.info(f"# kps detected per instance : {tabSpace.join(str(x) for x in np.arange(len(nKPDetectionCounts)) )}"); print(f"# kps detected per instance : {tabSpace.join(str(x) for x in np.arange(len(nKPDetectionCounts)) )}")
    logging.info(f"Counts for these kps        : {tabSpace.join(str(x) for x in nKPDetectionCounts)}"); print(f"Counts for these kps        : {tabSpace.join(str(x) for x in nKPDetectionCounts)}")



    #calculating top K accuracy - frame level
    try:
        n_blkSizes = len(gt_locs_list[0])

        k_topK = 16 #the k value in top K accuracies
        topK_accumulator = np.zeros((n_blkSizes, k_topK), int) #[[0]*k_topK]*n_blkSizes
        #pdb.set_trace()
        for n in range(n_blkSizes):
            gt_locs = [x[n] for x in gt_locs_list]
            for gt_loc in gt_locs:
                if gt_loc >= 0 and gt_loc < k_topK: #-1 indicates that the gt is not present in the cattlog dictionary
                    topK_accumulator[n, gt_loc]+=1
        topK_accuracies = np.cumsum(topK_accumulator, axis=1)/countsDict['total_allKPRulePass']

        printAndLog(f"\nTop K accuracies: (not overall! Of only the prediction stage = n imgs with top K acc / total_allKPRulePass)", printDebugInfoToScreen=True)
        printAndLog(f"{tabSpace.join([str(x) for x in list(range(1,k_topK+1))])}", printDebugInfoToScreen=True)

        for n in range(n_blkSizes):
            printAndLog(f"{tabSpace.join([f'{x:0.3f}' for x in topK_accuracies[n]])}", printDebugInfoToScreen=True)

        #logging.info(f"gt_locs_list = {gt_locs_list}")
        #print(f"gt_locs_list = {gt_locs_list}")
    except:
        printAndLog(f"\nTop K accuracies: (not overall! Of only the prediction stage = n imgs with top K acc / total_allKPRulePass) = 0 at all levels. There might be no correct cow identifications. ")


    #topK Image level accuracy - If inferring on image dataset
    if nImages != 0:
        topK_accuracies_imageLevel = np.cumsum(topK_accumulator, axis=1)/nImages

        printAndLog(f'\nTopK Accuracy at Image Level (on image dataset of {nImages} images)', printDebugInfoToScreen=True)
        for n in range(n_blkSizes):
            printAndLog(f"For blockSize index {n} ")
            printAndLog(f"{tabSpace.join(['k value']+[str(x) for x in list(range(1,k_topK+1))])}", printDebugInfoToScreen=True)
            printAndLog(f"{tabSpace.join(['Cumulative#:'] + [str(x) for x in np.cumsum(topK_accumulator, axis=1)[n]])}", printDebugInfoToScreen=True)
            printAndLog(f"{tabSpace.join(['Accuracy:'] + [f'{x:0.4f}' for x in topK_accuracies_imageLevel[n]])}", printDebugInfoToScreen=True)

    #overall accuracy - frame level
    overallAccuracy_frameLevel = countsDict['total_correct_preds'] / total_seenFrames
    printAndLog(f"\n(Ignore for Video Level Testing) Overall Frame Level Top1 Accuracy = total_correct_preds/totalFrames = {overallAccuracy_frameLevel}", printDebugInfoToScreen=True)

    #topK Video level accuracy - If inferring on video dataset
    if topK_VideoLevelDict is not None and nVideos != 0:

        if len(topK_VideoLevelDict) == 0:
            printAndLog(f"\nNo correct predictions. TopK accuracy = 0 for all K.", printDebugInfoToScreen=True)
        else:
            #print(f"topK_VideoLevelDict = {topK_VideoLevelDict}")
            topK_VideoLevelList = [0]*max(topK_VideoLevelDict)
            for k, v in topK_VideoLevelDict.items():
                topK_VideoLevelList[k-1] = v

            topK_accuracies_videoLevel = np.cumsum(topK_VideoLevelList)/nVideos

            printAndLog(f'\nTopK Accuracy at Video Level (on {nVideos} videos)', printDebugInfoToScreen=True)
            printAndLog(f"{tabSpace.join(['k value:']+[str(x) for x in list(range(1,len(topK_accuracies_videoLevel)+1))])}", printDebugInfoToScreen=True)
            printAndLog(f"{tabSpace.join(['Cumulative#:'] + [str(x) for x in np.cumsum(topK_VideoLevelList)])}", printDebugInfoToScreen=True)
            printAndLog(f"{tabSpace.join(['Accuracy:']+[f'{x:0.4f}' for x in topK_accuracies_videoLevel])}", printDebugInfoToScreen=True)

    # Cows with not a single correct top-1 instance level identification on any video frame.
    if cowsWithNoCorrectPreds_list is not None:
        printAndLog(f"\nThe following cows have no single correct top-1 instance level prediction on any video frame. They are not contributing to the top-k video level accuracy numbers.", printDebugInfoToScreen=True)
        if len(cowsWithNoCorrectPreds_list) == 0:
            printAndLog(f"<NIL>. Awesome! There are no such cows for this video set!", printDebugInfoToScreen=True)
        else:
            printAndLog(f"{cowsWithNoCorrectPreds_list}", printDebugInfoToScreen=True)

def draw_and_connect_keypoints(img, keypoints, keypoint_threshold=0.5, keypoint_names=False,
                               keypoint_connection_rules=False):
    """
    Copied from Detectron2 and modified by Manu

    Draws keypoints of an instance and follows the rules for keypoint connections
    to draw lines between appropriate keypoints. This follows color heuristics for
    line color.

    Args:
        keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
            and the last dimension corresponds to (x, y, probability).

    Returns:
        output (VisImage): image object with visualizations.
    """

    visible = {}
    # keypoint_names = self.metadata.get("keypoint_names")

    outImg = img.copy()

    for idx, keypoint in enumerate(keypoints):

        # draw keypoint
        x, y, prob = keypoint
        if prob > keypoint_threshold:

            outImg = cv2.circle(outImg, (round(x), round(y)), radius=5, color=(255, 0, 255), thickness=-1)

            # Manu
            # print(f"Keypoint Name = {keypoint_name}")
            # pdb.set_trace()

            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

                # MANU:
                # draw kp names
                font = cv2.FONT_HERSHEY_SIMPLEX;
                fontScale = 1;
                fontColor = (255, 0, 255);
                thickness = 1;
                lineType = 2
                cv2.putText(outImg, f"{keypoint_name}", (round(x) + 2, round(y) + 2), font, fontScale, fontColor,
                            thickness, lineType)

        if keypoint_connection_rules:
            for kp0, kp1, color in keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    outImg = cv2.line(outImg, (round(x0), round(y0)), (round(x1), round(y1)), color, thickness=4)

    return outImg

###################################################################################
from autoCattlogger.helpers.helper_for_morph import get_triangles
# if __name__ == "__main__":
#     from helper_for_morph import get_triangles
# else: #relative imports
#     from .helper_for_morph import get_triangles

import distinctipy #for selecting colors for triangles - https://pypi.org/project/distinctipy/

def draw_KP_partition_triangles(cowCroppedImg, keypoints, keypoint_threshold=0.5, keypoint_names=False,
                               keypoint_connection_rules=False, tri_colors=None, printDebugInfoToScreen=False):
    '''
    Draws triangles that partition the image. Corners points of the triangles are the given keypoints.

    :param cowCroppedImg: image of cow cropped with bounding rotated rectangle, with its neck up and tail down
    :param keypoints: keypoints in the form [[x,y,visibility score],[],[]....[]] - check autoCattlogger.py for more helo
    :param keypoint_threshold: thresh above which keypoints are considered visible
    :param keypoint_names:
    :param keypoint_connection_rules: Unused
    :param tri_colors: colors used for triangles [(R,G,B),....] where R,G,B are in [0,1] range
    :return: image with partition triangles overlaid, colors used to fill triangles (for using in another image of the same cow, as colors are generated at random evereytime distinctipy.get_color() is called)
    '''

    img = cowCroppedImg

    warpedKPs = keypoints
    visible = {}
    # keypoint_names = self.metadata.get("keypoint_names")

    outImg = img.copy()

    for idx, keypoint in enumerate(keypoints):

        # draw keypoint
        x, y, prob = keypoint
        if prob > keypoint_threshold:

            outImg = cv2.circle(outImg, (round(x), round(y)), radius=5, color=(255, 0, 255), thickness=-1)

            # Manu
            # print(f"Keypoint Name = {keypoint_name}")
            # pdb.set_trace()

            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

                # MANU:
                # draw kp names
                font = cv2.FONT_HERSHEY_SIMPLEX;
                fontScale = 1;
                fontColor = (255, 0, 255);
                thickness = 1;
                lineType = 2
                cv2.putText(outImg, f"{keypoint_name}", (round(x) + 2, round(y) + 2), font, fontScale, fontColor,
                            thickness, lineType)

        # if keypoint_connection_rules:
        #     for kp0, kp1, color in keypoint_connection_rules:
        #         if kp0 in visible and kp1 in visible:
        #             x0, y0 = visible[kp0]
        #             x1, y1 = visible[kp1]
        #             outImg = cv2.line(outImg, (round(x0), round(y0)), (round(x1), round(y1)), color, thickness=4)

    # ******* code copied from helper_for_morph.py>morph_cow_to_template function*************
    # define triangle connections for input KPs
    # define target KP locations - use dictionaries
    # define triangle connections for target KPS

    # remove the probability (visibility) values
    # if warpedKPs.shape[-1] == 3:
    tgt_KPs = np.zeros_like(warpedKPs)  # initialize
    KP_prbos = warpedKPs[:, -1]
    warpedKPs = warpedKPs[:, :-1]

    img_KP_locs = {k: v for k, v in zip(keypoint_names, warpedKPs)}  # dictionary

    img_h, img_w, _ = img.shape

    # these additional points must map to the same points after transformation
    additional_img_pt_locs = {'top_left_corner': (0, 0), 'top_right_corner': (img_w - 1, 0),
                              'bot_left_corner': (0, img_h - 1), 'bot_right_corner': (img_w - 1, img_h - 1),
                              'left_edge_center': (0, img_h // 2 - 1),
                              'right_edge_center': (img_w - 1, img_h // 2 - 1)}

    img_triangles = get_triangles(img_KP_locs, additional_img_pt_locs)

    if tri_colors is None:
        # generate colors
        tri_colors = distinctipy.get_colors(len(img_triangles)) #visually distinct colors

    selectedTriIds = [] # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # selecting only triagles inside the cow skeleton

    if selectedTriIds == []:
        selectedTriangles = list(img_triangles.items())
    else:
        selectedTriangles = [list(img_triangles.items())[triId - 1] for triId in selectedTriIds]  # -1 in triId-1 is because triId starts from 1 and list index starts from 0

    #remove later
    if printDebugInfoToScreen: print(f"\n\nIMG Shape (h,w) = ({img_h},{img_w})")
    if printDebugInfoToScreen: print(f"Selected Tris = {selectedTriangles}")
    if printDebugInfoToScreen: print(f"Tri = {np.array(selectedTriangles[0][-1]).reshape((-1,1,2)).astype(np.int32)}")
    #pdb.set_trace()


    tri_layer = np.zeros_like(img)

    # for tri, vals in tgt_triangles.items():
    for tri, vals in selectedTriangles:  # to select only a sequence of kps

        if printDebugInfoToScreen: print(f"Tri = {tri},  Vals = {vals}")
        tri_pts = np.array(vals).reshape((-1,1,2)).astype(np.int32) #np.float32(vals)
        #tri_pts = np.array([(y,x) for x, y in vals]).reshape((-1,1,2)).astype(np.int32) #np.float32(vals)
        if printDebugInfoToScreen: print(f"Contour points 1 = {tri_pts}")
        #tri_pts = np.array([[100,0], [0,100], [10,10]])
        #tri_pts = [np.array([[1,1],[10,50],[50,50]], dtype=np.int32)]

        tri_color = tuple([int(255*x) for x in tri_colors[tri-1]]) #(255,0,255) #tri_colors index starts from 0, triangle numbers start from 1 #tuple([int(255*x) for x in tri_colors[tri]])  #color of this triangle

        if printDebugInfoToScreen: print(f"Contour color = {tri_color}\n\n")

        cv2.drawContours(tri_layer, [tri_pts], 0, tri_color, -1)
        #cv2.drawContours(tri_layer, [tri_pts], 0, tri_color, 3)

    out_img = cv2.addWeighted(img, 0.2, tri_layer, 0.8, 0.0)
    #return tri_layer
    return out_img, tri_colors

def get_visible_KPs(keypoints, keypoint_names=False, keypoint_threshold=0.5):
    '''
    Returns dictionary of visible keypoints along with number of visible keypoints and total keypoints
    :param keypoints: (N,3) size array, N is the number of keypoints
    :param keypoint_names:
    :param keypoint_threshold: keypoint visibility lower threshold
    :return:
    '''

    visible = {}

    for idx, keypoint in enumerate(keypoints):

        x, y, prob = keypoint
        if prob > keypoint_threshold:

            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

    totalKPs = keypoints.shape[0]
    n_visibleKPs = len(visible.items())

    return visible, n_visibleKPs, totalKPs


def cow_tv_KP_rule_checker(keypoints, keypoint_connection_rules=False, keypoint_names=False, keypoint_threshold=0.5, cowW=350, cowH=750, frameW=1920, frameH=1080):
    '''
    Function to check whether the detected keypoints form a valid cow or not.
    This should return a number that is proportional to validity.

    use the returned maxConfidence value to help you decide the confidence threshold
    you can set the confidence threshold relative to the returned max threshold
    eg: if confidence > maxConfidence - 1: do...

    :param keypoints: (N,3) size array, N is the number of keypoints
    :param keypoint_connection_rules:
    :param keypoint_names:
    :param cowW: Width of the cow bbox in top view
    :param cowH: height of cow bbox in top view (actually the length of the cow)
    :param frameW: width of the frame used for detection
    :param frameH: height of the frame used for detection
    :return: confidence (a value between 0 and the returned max confidence),

    '''

    #not using frameW and frameH to get relative lengths (fractional lengths) of the distances between cow KPs
    #useing cowW and cowH to get relative lengths (fractional lengths) of the distances between cow KPs
    lengthDenominator =  np.sqrt(cowW*cowH) #np.sqrt(frameW*frameH)

    visible = {}

    for idx, keypoint in enumerate(keypoints):

        x, y, prob = keypoint
        if prob > keypoint_threshold:

            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

    #distances = {}
    #for kp0, kp1, color in keypoint_connection_rules:
    #    if kp0 in visible and kp1 in visible:
    #        x0, y0 = visible[kp0]
    #        x1, y1 = visible[kp1]
    #
    #        distance = np.linalg.norm(np.array([x0,y0]) - np.array([x1, y1])) #l2 norm for now
    #        distances[(kp0, kp1)] = distance


    confidence = 0
    maxConfidence = 0 #put in the total number of checks you wish to perform here

    allVisible = False
    if len(visible.items()) == 10: #all keypoints are visible
        allVisible = True

    ang_deviation_margin = 0.25 # in 0 to 1
    ang_deviation_margin_2 = 0.45 #0.40 # in 0 to 1 #for a few angles
    ang_deviation_margin_3 = 0.60 #0.40 # in 0 to 1 #for angle between shoulders and withers @ center_back
    len_deviation_margin = 0.25 # in 0 to 1
    len_deviation_margin_2 = 0.375 #0.35 # in 0 to 1 #for a few lengths

    angThresh_lpin_hc_rpin = 25
    angThresh_lhip_hc_rhip = 90
    angThresh_lshld_cback_rshld = 40

    #these fractional length (f-lenght or flen) thresholds are all lower limits
    flenThresh_hips_hipCon = 145/lengthDenominator
    flenThresh_shldrs_withers = 90/lengthDenominator
    flenThresh_pins_tailHead = 50/lengthDenominator
    flenThresh_withers_hipCon = 500/lengthDenominator #you might need to change this to accomodate bent cows
    flenThresh_hipCon_tailHead = 200/lengthDenominator

    #if 'left_pin_bone' in visible and 'right_pin_bone' in visible and 'tail_head' in visible and 'hip_connector' in visible:
    if allVisible:
        #for now, proceeding only if all keypoints are visible

        # ******************************************************************************************#
        # ***********************   CHECKING FRACTIONAL LENGTH REQUIREMENTS ************************#

        #FOR REFERENCE
        #keypoint names are  ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]

        #measuring the fractional lengths
        flen_hips_hipCon = min(*get_length_deviation(('hip_connector', 'left_hip_bone'), ('hip_connector', 'right_hip_bone'), visible=visible)[-2:]) / lengthDenominator
        flen_shldrs_withers = min(*get_length_deviation(('left_shoulder', 'withers'), ('withers', 'right_shoulder'), visible=visible)[-2:]) / lengthDenominator
        flen_pins_tailHead = min(*get_length_deviation(('left_pin_bone', 'tail_head'), ('tail_head', 'right_pin_bone'), visible=visible)[-2:]) / lengthDenominator
        flen_withers_hipCon = min(*get_length_deviation(('withers', 'hip_connector'), ('withers', 'hip_connector'), visible=visible)[-2:]) / lengthDenominator # I know this does double calculations - but it fits the template followed above, lazy to change
        flen_hipCon_tailHead = min(*get_length_deviation(('hip_connector', 'tail_head'), ('hip_connector', 'tail_head'), visible=visible)[-2:]) / lengthDenominator # I know this does double calculations - but it fits the template followed above, lazy to change

        #minimum length hips - hip con
        maxConfidence += 1
        if flen_hips_hipCon >= flenThresh_hips_hipCon:
            confidence += 1
        else:
            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Min fractional Length of distance between hip bones and hip connector is too small {flen_hips_hipCon} < {flenThresh_hips_hipCon}")

        #minimum flength shoulders - withers
        maxConfidence += 1
        if flen_shldrs_withers >= flenThresh_shldrs_withers:
            confidence += 1
        else:
            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Min fractional Length of distance between shoulders and withers is too small {flen_shldrs_withers} < {flenThresh_shldrs_withers}")

        #minimum flength pin bones - tail head
        maxConfidence += 1
        if flen_pins_tailHead >= flenThresh_pins_tailHead:
            confidence += 1
        else:
            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Min fractional Length of distance between pin bones and tail head is too small {flen_pins_tailHead} < {flenThresh_pins_tailHead}")

        #minimum flength withers - hip connector
        maxConfidence += 1
        if flen_withers_hipCon >= flenThresh_withers_hipCon:
            confidence += 1
        else:
            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Min fractional Length of distance between withers and hip connector is too small {flen_withers_hipCon} < {flenThresh_withers_hipCon}")

        #minimum flength hip connector - tail head
        maxConfidence += 1
        if flen_hipCon_tailHead >= flenThresh_hipCon_tailHead:
            confidence += 1
        else:
            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Min fractional Length of distance between hip connector and tail head is too small {flen_hipCon_tailHead} < {flenThresh_hipCon_tailHead}")

        #******************************************************************************************#
        #***************************    PIN BONES AND HIP CONNECTOR *******************************#

        # check if angle and distance between pin bones and tail head at hip connector are within a margin
        ang_deviation, ang0, ang1 = get_ang_deviation(('left_pin_bone', 'hip_connector', 'tail_head'), ('right_pin_bone', 'hip_connector', 'tail_head'), visible=visible)
        #checking angle deviation
        maxConfidence += 1
        #if ang_deviation < ang_deviation_margin:
        if ang_deviation < ang_deviation_margin_2: #increased allowed deviation margin
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high ang dev - pin bones and tail head @ hip con |> dev:{ang_deviation} >= margin:{ang_deviation_margin_2} ")

        len_deviation, len0, len1 = get_length_deviation(('hip_connector', 'left_pin_bone'), ('hip_connector', 'right_pin_bone'), visible=visible)

        #checking length deviation
        maxConfidence += 1
        if len_deviation < len_deviation_margin:
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high len dev - hip con and pin bones |> dev:{len_deviation} >= margin:{len_deviation_margin} ")

        #also check if lpin-hcon-rpin angle is above a threshold (to eliminate cases where pin bones are detected at the same point)
        #ang_lpin_hc_rpin = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail! (See 5 June 2022 Journal log)
        _, ang0, _ = get_ang_deviation(('left_pin_bone', 'hip_connector', 'right_pin_bone'), ('left_pin_bone', 'hip_connector', 'right_pin_bone'), visible=visible)
        ang_lpin_hc_rpin = ang0
        maxConfidence += 1
        if ang_lpin_hc_rpin > angThresh_lpin_hc_rpin:
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Angle between pin bones at hip connector is too small {ang_lpin_hc_rpin} <= {angThresh_lpin_hc_rpin}")

        #******************************************************************************************#
        #***************************    HIP BONES AND HIP CONNECTOR *******************************#

        # check if angle and distance between hip bones and tail head at hip connector are within a margin
        #this is the rigid part of the cow's body, so our constraints can be rigid
        ang_deviation, ang0, ang1 = get_ang_deviation(('left_hip_bone', 'hip_connector', 'tail_head'), ('right_hip_bone', 'hip_connector', 'tail_head'), visible=visible)
        maxConfidence += 1
        if ang_deviation < ang_deviation_margin:
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high ang dev - hip bones and tail head @ hip con |> dev:{ang_deviation} >= margin:{ang_deviation_margin} ")

        len_deviation, len0, len1 = get_length_deviation(('hip_connector', 'left_hip_bone'), ('hip_connector', 'right_hip_bone'), visible=visible)
        maxConfidence += 1
        if len_deviation < len_deviation_margin:
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high len dev - hip con and hip bones |> dev:{len_deviation} >= margin:{len_deviation_margin} ")

        #also check if the left hip - hip_con - right hip angle > threshold (to prevent all 3 to be detected at the same point)
        #ang_lhip_hcon_rhip = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail
        _, ang0, _ = get_ang_deviation(('left_hip_bone', 'hip_connector', 'right_hip_bone'), ('left_hip_bone', 'hip_connector', 'right_hip_bone'), visible=visible)
        ang_lhip_hcon_rhip = ang0
        maxConfidence += 1
        if ang_lhip_hcon_rhip > angThresh_lhip_hc_rhip:
            confidence+=1
            #logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Pass: Angle between hip bones at hip connector is too small {ang_lhip_hcon_rhip} > {angThresh_lhip_hc_rhip}") #delete this line later
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Angle between hip bones at hip connector is too small {ang_lhip_hcon_rhip} <= {angThresh_lhip_hc_rhip}")

        #******************************************************************************************#
        #***************************    SHOULDERS AND WITHERS *******************************#

        #check if angle and distance between left and right shoulders with withers at center back are within a margin
        ang_deviation, ang0, ang1 = get_ang_deviation(('left_shoulder', 'center_back', 'withers'),
                                                   ('right_shoulder', 'center_back', 'withers'), visible=visible)
        maxConfidence += 1
        #if ang_deviation < ang_deviation_margin: #YOU MIGHT HAVE TO CHANGE THIS MARGIN IF COW IS NOT SEEN EXACTLY TOP DOWN
        if ang_deviation < ang_deviation_margin_3: #changing margin
             confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high ang dev - shoulders and withers @ center_back |> dev:{ang_deviation} >= margin:{ang_deviation_margin_3} ")

        len_deviation, len0, len1 = get_length_deviation(('center_back', 'left_shoulder'),
                                                     ('center_back', 'right_shoulder'), visible=visible)
        maxConfidence += 1
        #if len_deviation < len_deviation_margin:
        if len_deviation < len_deviation_margin_2: #changing margin
             confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: high len dev - center_back and shoulders |> dev:{len_deviation} >= margin:{len_deviation_margin_2} ")

        #also check if the lshoulder - center back - rshoulder angle > threshold (to eliminate cases where both shoulders and wihters to be detected at the same point)
        #ang_lshld_cback_rshld = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail - adding two angles with abs will give positive result even if shouder points are detected at the same location
        _, ang0, _ = get_ang_deviation(('left_shoulder', 'center_back', 'right_shoulder'), ('left_shoulder', 'center_back', 'right_shoulder'), visible=visible) #get the angle directly - again - double computation, I know - cut me some slack
        ang_lshld_cback_rshld = ang0
        maxConfidence += 1
        if ang_lshld_cback_rshld >= angThresh_lshld_cback_rshld:
            confidence += 1
            #logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Pass: Angle between shoulders at center back {ang_lshld_cback_rshld}") #remove later
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Angle between shoulders at center back is too small {ang_lshld_cback_rshld} <= {angThresh_lshld_cback_rshld}")


        #******************************************************************************************#
        #***************************    SHOULDERS AND PIN BONES *******************************#

        #check if shoulders and pin bones are not detected on the same side of the cow
        #to do this, check if angle made by left shoulder and left pin bone at center back is > 90 derees, do the same for right shoulder and right pin bone
        _, ang0, ang1 = get_ang_deviation(('left_shoulder', 'center_back', 'left_pin_bone'),
                                                      ('right_shoulder', 'center_back', 'right_pin_bone'), visible=visible)
        maxConfidence += 2
        if ang0 > 90: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Left shoulder and left pin bone too close to each other |> angle: {ang0} < margin: {90}")

        if ang1 > 90: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Right shoulder and right pin bone too close to each other |> angle: {ang1} < margin: {90}")


        #check if pin bones and hip bones are not detected close to each other
        #do this by checking if the angle between pin bones and hip bones at the hip connector is > than a threshold
        _, ang0, ang1 = get_ang_deviation(('left_hip_bone', 'hip_connector', 'left_pin_bone'),
                                                      ('right_hip_bone', 'hip_connector', 'right_pin_bone'), visible=visible)
        maxConfidence += 2
        if ang0 > 40: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Left hip bone and left pin bone too close to each other wrt center_back angle |> angle: {ang0} > margin: {40}")

        if ang1 > 40: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Right hip bone and right pin bone too close to each other wrt center_back angle |> angle: {ang1} > margin: {40}")

        #******************************************************************************************#
        #***************************    TAIL-HEAD AND WITHERS *******************************#

        #check if tail head and withers are not detected at the same point (or anywhere very close to each other)
        #We do this in 3 steps,
        #   - check if angle between center back and tail head at hip connector is above a certain angle
        #   - check if angle between withers and hip connector at center back is above a certain angle
        #   - check if angle between withers and tail head at hip connector is above a certain angle
        _, ang0, ang1 = get_ang_deviation(('center_back', 'hip_connector', 'tail_head'),
                                                      ('withers', 'center_back', 'hip_connector'), visible=visible)
        _, ang2, _    = get_ang_deviation(('withers', 'hip_connector', 'tail_head'), ('withers', 'hip_connector', 'tail_head'), visible=visible)
        maxConfidence += 3
        if ang0 > 60: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Center_back and tail head too close to each other |> angle@hip_con: {ang0} > margin: {60}")

        if ang1 > 60: #it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Withers and hip_con too close to each other |> angle@center_back: {ang1} > margin: {60}")

        if ang2 > 90: # it considers the inner angle
            confidence += 1
        else: logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: Withers and tail head too close to each other |> angle@hip_con: {ang2} > margin: {60}")


    return confidence, maxConfidence

def cow_tv_KP_rule_checker_auxiliary(keypoints, keypoint_names, maskContour=None, printDebugInfoToScreen=False):
    '''
    Not a replacement for rule checkers 1 and 2. Checks for a different set of rules.
    To be used to filter hard examples - for self supervision.

    NOTE: ALL KEYPOINTS ARE SUPPOSED TO BE VISIBLE FOR THIS TO WORK

    :param keypoints: (N,3) size array, N is the number of keypoints
    :param keypoint_names: List of names of keypoints (lowercase, multiple words separated by underscore. Eg: center_back, left_shoulder,..."
    :param maskContour: opencv contour object of the instance mask
    :return:
    '''

    maxConfidence = 0
    confidence = 0

    visible_KPs, n_visibleKPs, totalKPs = get_visible_KPs(keypoints=keypoints, keypoint_names=keypoint_names, keypoint_threshold=0.5)

    if totalKPs == n_visibleKPs:

        #we check if the spine is straight
        maxConfidence += 1
        p1 = x1, y1 = visible_KPs['tail_head']
        p2 = x2, y2 = visible_KPs['hip_connector']
        p3 = x3, y3 = visible_KPs['center_back']
        p4 = x4, y4 = visible_KPs['withers']

        n = np.array([1, 1, 2, 1])
        a, b, c = curve = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], w=np.sqrt(n), deg=2)  # ax^2 + bx + c

        if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
        x_pw2_coeff_ulim = 1e-4 #12e-5  # 1e-4 #upper limit of x^2's coefficient # not using any limits for now

        # we need the spine line to be straight so that the reflected point doesn't go out of the cow or comes inside the cow
        if a <= x_pw2_coeff_ulim:
            confidence+=1

    return confidence, maxConfidence


def cow_tv_KP_rule_checker2(keypoints, datasetKpStatsDict, maskContour=None, keypoint_connection_rules=False, keypoint_names=False, keypoint_threshold=0.5, cowW=350, cowH=750, frameW=1920, frameH=1080, returnInstanceKpStatsDict=False, printDebugInfoToScreen=False):
    '''
    Meant to do the same function as cow_tv_KP_rule_checker. But this one uses code from helper_for_stats.py and threshold values determined from the human annotations.

    :param keypoints: (N,3) size array, N is the number of keypoints
    :param datasetKpStatsDict: kp stats dict of the train dataset - {'minStatDict':<statsDict>, 'maxStatDict':<statsDict>} -read this from output/statOutputs/<>.p file and pass it
                               This code is not loading the file again and again as it is a time waste. Load it in the calling code and then pass it here.
    :param maskContour: OpenCV contour object of the cow top view mask. (Array of polygon points.) - will use this to check keypoint nearness to mask edges.
    :param keypoint_connection_rules:
    :param keypoint_names: list of names of keypoints in order
    :param keypoint_threshold: the threshold lower bound for keypoint to be considered visible - usually 0.5
    :param cowW: Width of the cow bbox in top view
    :param cowH: height of cow bbox in top view (actually the length of the cow)
    :param frameW: width of the frame used for detection
    :param frameH: height of the frame used for detection
    :param returnInstanceKpStatsDict: If True, returns instanceKpStatsDict. To be used for debugging.
    :return: confidence (a value between 0 and the returned max confidence)
    '''

    dsObj1 = DatasetStats()
    instanceKpStatsDict = dsObj1.computeKPRuleStats_perInstance(keypoints=keypoints, cowW=cowW, cowH=cowH, keypoint_connection_rules=keypoint_connection_rules,keypoint_threshold=keypoint_threshold, keypoint_names=keypoint_names, allKPs_mustBeVisible = True,  maskContour = maskContour, frameW=1920, frameH=1080)
    minStatDict = datasetKpStatsDict['minStatDict']
    maxStatDict = datasetKpStatsDict['maxStatDict']

    confidence = 0
    maxConfidence = 0
    instanceRulePassesDict = {} # a python dictionary that contains pass/fail values for each rule, has 'NA' if rule is ignored - to be used to predict misplaced keypoints

    def compareDicts(instanceDict, modelDict, comparisonFn = lambda inst, model : inst > model,  fnMessage =">", ignoreNDict=None, confidence=0, maxConfidence=0, printDebugInfoToScreen=printDebugInfoToScreen):
        '''
        A function that compares nested dictionaries. Uses the ndicts package.

        :param instanceDict: the kp stats dict obtained from the cow instance
        :param modelDict: the kp stats dict from the ground truth/annotation
        :param comparisonFn: a function that takes in the two values and returns a boolean value. The rule is passed if the returned value is True. This can be as simple as a lambda function lambda x,y : x>=y
        :param fnMessage: message that needs to be logged in case of rule failure. Format: {instanceValue} {fnMessage} {modelValue} eg: 2 < 3
        :param ignoreNDict: nested dictionary object with Trues in places where the comparisons need to be ignored
        :param confidence: the initial confidence value (use this if you are comparing multiple nested dicts, one after the other- eg: minDict, maxDict)
        :param maxConfidence: the initial maxConfidence value (use this if you are comparing multiple nested dicts, one after the other- eg: minDict, maxDict)
        :return: new confidence and maxConfidence values
        '''
        pass

        # https://github.com/edd313/ndicts/blob/main/tutorials/NestedDict_tutorial.ipynb
        # https://edd313.github.io/ndicts/nested_dict/
        # https://pypi.org/project/ndicts/
        instNDict = NestedDict(instanceDict)
        modelNDict = NestedDict(modelDict)

        instRulePassesNDict = NestedDict({}) #a nested dictionary that contains pass/fail values for each rule, has 'NA' if rule is ignored - to be used to predict misplaced keypoints

        #for instItem, modelItem in zip(instNDict.items(), modelNDict.items()): #doing so would iterate both the dict keys in different orders!!!
        for instK, instV in instNDict.items():
            #print(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> instItem = {instItem}")
            #logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> instItem = {instItem}")


            #instK, instV = instItem
            #modelK, modelV = modelItem
            #ignoreV = ignoreNDict[instK]

            modelK = instK
            modelV = modelNDict[modelK]
            ignoreV = ignoreNDict[instK]

            #pdb.set_trace()

            #if printDebugInfoToScreen: print(f"Values for comparison {instK}: InstV:{instV}, ModelV:{modelV}") #the keys are the same for instance and model
            #logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Values for comparison {instK}: InstV:{instV}, ModelV:{modelV}") #the keys are the same for instance and model

            if not ignoreV:
                maxConfidence += 1
                if comparisonFn(instV, modelV):
                    confidence+=1
                    instRulePassesNDict[instK] = 'Pass'
                else:

                    instRulePassesNDict[instK] = 'Fail'

                    if printDebugInfoToScreen: print(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: {instK} InstanceVal = {instV} {fnMessage} modelVal = {modelV} of modelK = {modelK}")
                    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> KP Rule Fail: {instK} InstanceVal = {instV} {fnMessage} modelVal = {modelV} of modelK = {modelK}")

                    #pdb.set_trace()
            else:
                instRulePassesNDict[instK] = 'NA'

        instRulePassesDict = instRulePassesNDict.to_dict() #converting to normal python dictionary

        return confidence, maxConfidence, instRulePassesDict

    mode = 2 #mode = 1

    if instanceKpStatsDict != None: #it is None if all visible is required but all keypoints are not visible.
        # first let us try the simple approach
        # the obtained values must all be lesser than the max values and must all be more than the min values
        # NOTE: YOU NEED TO MAKE SURE THAT THE MIN MAX DATASET STAT DICT YOU SUPPLY HAS THE SAME KEYS AS THOSE YOU GENERATE FOR EACH INSTANCE IN THIS FUNCTION
        # IF YOUR OLD DATASET STAT DICTIONARY IS OUTDATED, REGENERATE IT - helper_for_stat.py/<DatasetStats_obj>.computeKPRuleStats_onDataset()

        #should add logging ability to this!

        #we may have to throw away the curve parameters later

        ignoreNDictMin = NestedDict(instanceKpStatsDict).copy()
        for k, v in ignoreNDictMin.items():
            ignoreNDictMin[k] = True
        ignoreNDictMax = ignoreNDictMin.copy()

        ignoreNDictMin[('fractional_lengths', 'flen_hips_hipCon')] = ignoreNDictMin[('fractional_lengths', 'flen_shldrs_withers')] = ignoreNDictMin[('fractional_lengths', 'flen_pins_tailHead')] = ignoreNDictMin[('fractional_lengths', 'flen_withers_hipCon')] = ignoreNDictMin[('fractional_lengths', 'flen_hipCon_tailHead')] = False
        ignoreNDictMin[('pinBones_hcon', 'angles', 'ang_lpin_hcon_rpin')] = ignoreNDictMin[('hipBones_hcon', 'angles', 'ang_lhip_hcon_rhip')] = ignoreNDictMin[('shldrs_withers', 'angles', 'ang_lshldr_cback_rshldr')] = False
        ignoreNDictMin[('shldrs_pinBones', 'angles', 'ang_lshldr_cback_lpin')] = ignoreNDictMin[('shldrs_pinBones', 'angles', 'ang_rshldr_cback_rpin')] = ignoreNDictMin[('hipBones_pinBones', 'angles', 'ang_lhip_hcon_lpin')] = ignoreNDictMin[('hipBones_pinBones', 'angles', 'ang_rhip_hcon_rpin')] = ignoreNDictMin[('spinePoints', 'angles', 'ang_cback_hcon_thead')] = ignoreNDictMin[('spinePoints', 'angles', 'ang_witr_cback_hcon')] = ignoreNDictMin[('spinePoints', 'angles', 'ang_witr_hcon_thead')] = False
        ignoreNDictMax[('pinBones_hcon', 'angles', 'angDev_pinBones_thead_at_hcon')] = ignoreNDictMax[('pinBones_hcon', 'lengths', 'lenDev_pinBones_hcon')] = ignoreNDictMax[('hipBones_hcon', 'angles', 'angDev_hipBones_thead_at_hcon')] = ignoreNDictMax[('hipBones_hcon', 'lengths', 'lenDev_hipBOnes_hcon')] = ignoreNDictMax[('shldrs_withers', 'angles', 'angDev_shldrs_withers_at_cback')] = ignoreNDictMax[('shldrs_withers', 'lengths', 'lenDev_shldrs_cback')] = False

        #remove later
        #ignore nothing
        for k, v in ignoreNDictMin.items():
            ignoreNDictMin[k] = False
        ignoreNDictMax = ignoreNDictMin.copy()

        if mode == 1:
            minStatsList = [x for x in dsObj1.flattenDict(minStatDict) if type(x) is not str]
            maxStatsList = [x for x in dsObj1.flattenDict(maxStatDict) if type(x) is not str]
            instanceStatsList = [x for x in dsObj1.flattenDict(instanceKpStatsDict) if type(x) is not str]

            
            maxConfidence = len(instanceStatsList)

            comparator = list(zip(instanceStatsList, minStatsList, maxStatsList)) #for testing
            if printDebugInfoToScreen: print(f"\ncomparator values = {comparator}")
            rulePasses = [(instVal >= minVal and instVal <= maxVal) for (instVal, minVal, maxVal) in zip(instanceStatsList, minStatsList, maxStatsList)]
            confidence = sum(rulePasses)

        elif mode == 2:
            #remove if they do not work
            #to add some margin to the GT values in hopes of increasing overall accuracy
            lb_prop = 1 #0                            #0.25 -works but avoid #0.5 #-worked #0.75 #-worked #0.90 #=1 #lower bound proportion
            ub_prop = 1 #(0,10) -passes just anything #1.75 -works but avoid #1.5 #-worked #1.25 #-worked #1.10 #=1 #upper bound proportion

            confidence, maxConfidence, instanceMinRulePassesDict = compareDicts(instanceDict=instanceKpStatsDict, modelDict=minStatDict, comparisonFn=lambda x,y:x>=y*lb_prop, fnMessage="!>=", ignoreNDict=ignoreNDictMin, confidence=confidence, maxConfidence=maxConfidence)
            confidence, maxConfidence, instanceMaxRulePassesDict = compareDicts(instanceDict=instanceKpStatsDict, modelDict=maxStatDict, comparisonFn=lambda x,y:x<=y*ub_prop, fnMessage="!<=", ignoreNDict=ignoreNDictMax, confidence=confidence, maxConfidence=maxConfidence)
            instanceRulePassesDict['minStatDict'] = instanceMinRulePassesDict
            instanceRulePassesDict['maxStatDict'] = instanceMaxRulePassesDict

    if returnInstanceKpStatsDict:
        return confidence, maxConfidence, instanceRulePassesDict, instanceKpStatsDict
    else:
        return confidence, maxConfidence, instanceRulePassesDict



def get_length_deviation(pts0, pts1, visible):
    '''

    :param pts0: tuple with names of the two keypoints forming the line segment
    :param pts1: tuple with names of the two keypoints forming the line segment
    :param visible: the visible keypoints dictionary.
    :return: (deviation, len0, len1)
    '''

    pt00, pt01 = pts0
    pt10, pt11 = pts1

    vec0 = np.array(visible[pt01]) - np.array(visible[pt00])
    vec1 = np.array(visible[pt11]) - np.array(visible[pt10])

    l1 = np.linalg.norm(vec0)
    l2 = np.linalg.norm(vec1)
    #len_deviation = np.abs(l1 - l2) / max(l1, l2)
    len_deviation = np.abs(l1 - l2) / (max(l1, l2) + 1e-10) #to avoid nans on div by 0 - happens if lenght is 0 - which itself is a case of erroneous kp detection)

    return len_deviation, l1, l2


def get_ang_deviation(pts0, pts1, visible):
    '''
    visible: the visible keypoints dictionary

    pts1 and pts2 are tuples with names of 3 keypoints forming the angle.
    angle is formed at second keypoint (the centre one)
    :return: (deviation, angle0, angle1)
    '''
    pt00, pt01, pt02 = pts0
    pt10, pt11, pt12 = pts1

    vec00 = np.array(visible[pt00]) - np.array(visible[pt01]);
    vec01 = np.array(visible[pt02]) - np.array(visible[pt01]);

    vec10 = np.array(visible[pt10]) - np.array(visible[pt11]);
    vec11 = np.array(visible[pt12]) - np.array(visible[pt11]);

    #ang0 = np.arccos(np.dot(vec00, vec01) / (np.linalg.norm(vec00) * np.linalg.norm(vec01))) / np.pi * 180
    #ang1 = np.arccos(np.dot(vec10, vec11) / (np.linalg.norm(vec10) * np.linalg.norm(vec11))) / np.pi * 180

    ang0 = np.abs(np.arctan2(vec01[1], vec01[0]) - np.arctan2(vec00[1], vec00[0])) / np.pi * 180
    ang1 = np.abs(np.arctan2(vec11[1], vec11[0]) - np.arctan2(vec10[1], vec10[0])) / np.pi * 180

    #to always return the smaller (inner angle), positive angle
    ang0 = min(abs(ang0), 360 - abs(ang0))
    ang1 = min(abs(ang1), 360 - abs(ang1))

    #ang_deviation = np.abs(ang0 - ang1) / max(ang0, ang1)
    ang_deviation = np.abs(ang0 - ang1) / (max(ang0, ang1)+ 1e-10) #to avoid nans on div by 0 - happens if angle is 0)

    return (ang_deviation, ang0, ang1)



if __name__ == "__main__":
    img = cv2.imread("../../Data/CowTopView/for_KP_cattlog/june_09_2022/cow_topView_060922_images_datelessImgNames/images_test_datelessImgNames/6129_12.jpg") #note that this is not a cropped image
    keypoints = np.array([1222, 392, 2, 1400, 547, 2, 1286, 673, 2, 1069, 621, 2, 519, 516, 2, 547, 756, 2, 666, 918, 2, 200, 765, 2, 230, 853, 2, 270, 922, 2]).reshape((-1,3))
    kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector",
           "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]
    outImg = draw_KP_partition_triangles(img, keypoints, keypoint_names=kpn)
    cv2.imshow('triangles', cv2.resize(outImg, (640,360)))
    cv2.waitKey(0)
    pass