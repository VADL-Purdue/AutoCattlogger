"""
Author: Manu Ramesh
To get the stats of the cow datasets


"""
import os, glob, pickle, pdb, yaml
import numpy as np, cv2
from pycocotools.coco import COCO
import pandas as pd
import logging, inspect
from shapely.geometry import Point, Polygon

try:
    from ndicts.ndicts import DataDict, NestedDict #https://github.com/edd313/ndicts for comparing nested dictionaries
except ImportError:
    from ndicts import DataDict, NestedDict  # https://github.com/edd313/ndicts for comparing nested dictionaries

class DatasetStats():

    def __init__(self, dsetFilePath=None, img_dir=None):
        self.dsetFilePath = dsetFilePath
        if self.dsetFilePath is not None: #it can be none if you are using this class elsewhere, eg: in _autoCattloggerBase.py
            self.dsetName = self.dsetFilePath.split('/')[-1].split('.')[0]
        self.coco = COCO(self.dsetFilePath)
        self.img_dir = img_dir

    def get_cowW_cowH(self, mask_img):
        # returns cow widht and height
        # adapted from _autoCattloggerBase.py

        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            print(f"More than one contour detected. Not Breaking. Using largest contour.")
            contours = sorted(contours, key=cv2.contourArea)[::-1]  # sort in descending order
            # cv2.imwrite(f"multi_contour_image_{frameCount}.jpg", overlap)
            # break

        cnt = contours[0]

        rect = cv2.minAreaRect(cnt)
        boxAngle = rect[-1]  # angle that rectangle makes with horizontal, in degrees
        box = cv2.boxPoints(rect)  # 4 x 2 np array [[xA, yA], [xB, yB], [xC, yC], [xD, yD]
        box = np.int0(box)
        boxW, boxH = rect[1];
        boxW = int(boxW);
        boxH = int(boxH)

        print(f"\nboxW = {boxW}, boxH = {boxH}\n")

        # use distance formula
        lenAB = int(np.sqrt((box[1, 0] - box[0, 0]) ** 2 + (box[1, 1] - box[0, 1]) ** 2))
        lenBC = int(np.sqrt((box[2, 0] - box[1, 0]) ** 2 + (box[2, 1] - box[1, 1]) ** 2))

        cowW = min(lenAB, lenBC)
        cowH = max(lenAB, lenAB) #cow Height is actually the length of the cow.

        return cowW, cowH

    def computeKPRuleStats_perInstance(self,keypoints, cowW, cowH, keypoint_connection_rules=False,keypoint_threshold=0.5, keypoint_names=False, allKPs_mustBeVisible = False, maskContour = None, frameW=1920, frameH=1080):
        '''
        Computes stat values for KP rules for a given instance
        This function is to be used as the base function for computing KP Rule stats for the entire dataset and for the rule checker function in helper_for_infer.py
        (this is yet to be incorporated in helper for infer.py)

        This function should work even if all keypoints are not visible. This way, we can be sure to set the correct threshold limits, making these checks future proof - in case you want to try to guess cowID even if all keypoints are not visible.

        :param allKPs_mustBeVisible: if True, returns a single None if all KPs are not visible. If false, returns a stat dict with Nones in place of missing items.

        '''

        # not using frameW and frameH to get relative lengths (fractional lengths) of the distances between cow KPs
        # useing cowW and cowH to get relative lengths (fractional lengths) of the distances between cow KPs
        f_cowW = cowW/frameW*1920; f_cowH = cowH/frameH*1080 #fractional cow widhts, scaled to 1920x1080. Remove the 1920 and 1080 scaling later once everything works.
        lengthDenominator = np.sqrt(f_cowW * f_cowH).item() # saving as builtin data type as numpy types are not saved properly in YAML files  # np.sqrt(frameW*frameH)

        kpStatsDict = {}

        visible = {}

        for idx, keypoint in enumerate(keypoints):

            x, y, prob = keypoint
            if prob > keypoint_threshold:

                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)

        allVisible = False
        if len(visible.items()) == 10:  # all keypoints are visible
            allVisible = True

        if (allKPs_mustBeVisible and allVisible) or not allKPs_mustBeVisible:
            # for now, proceeding only if all keypoints are visible

            # ******************************************************************************************#
            # ********************  FOR CHECKING FRACTIONAL LENGTH REQUIREMENTS ************************#

            # FOR REFERENCE
            # keypoint names are  ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]

            # measuring the fractional lengths
            l1, l2 = self.get_length_deviation(('hip_connector', 'left_hip_bone'), ('hip_connector', 'right_hip_bone'), visible=visible)[-2:]
            flen_lhip_hipCon = l1 / lengthDenominator if (l1 is not None) else None #all(temp) is Ture if no element of Temp is None
            flen_rhip_hipCon = l2 / lengthDenominator if (l2 is not None) else None #all(temp) is Ture if no element of Temp is None

            l1,l2 = self.get_length_deviation(('left_shoulder', 'withers'), ('withers', 'right_shoulder'), visible=visible)[-2:]
            flen_lshldr_withers = l1 / lengthDenominator if (l1 is not None) else None #all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None
            flen_rshldr_withers = l2 / lengthDenominator if (l2 is not None) else None #all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None

            l1,l2 = self.get_length_deviation(('left_pin_bone', 'tail_head'), ('tail_head', 'right_pin_bone'), visible=visible)[-2:]
            flen_lpin_tailHead = l1 / lengthDenominator if (l1 is not None) else None #if not all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None
            flen_rpin_tailHead = l2 / lengthDenominator if (l2 is not None) else None #if not all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None

            l1, _ = self.get_length_deviation(('withers', 'hip_connector'), ('withers', 'hip_connector'), visible=visible)[-2:]
            flen_withers_hipCon = l1 / lengthDenominator if (l1 is not None) else None #if not all(t is None for t in temp) else None  # I know this does double calculations - but it fits the template followed above, lazy to change #all(temp) is Ture if no element of Temp is None

            l1, l2 = self.get_length_deviation(('hip_connector', 'tail_head'), ('withers', 'center_back'), visible=visible)[-2:] # Do not consider length deviation here, they measure two different distance!
            flen_hipCon_tailHead = l1 / lengthDenominator if (l1 is not None) else None #if not all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None
            flen_withers_cback = l2 / lengthDenominator if (l2 is not None) else None #if not all(t is None for t in temp) else None #all(temp) is Ture if no element of Temp is None

            kpStatsDict["fractional_lengths"] = {"flen_lhip_hipCon":flen_lhip_hipCon,"flen_rhip_hipCon":flen_rhip_hipCon,
                                                 "flen_lshldr_withers":flen_lshldr_withers, "flen_rshldr_withers":flen_rshldr_withers,
                                                 "flen_lpin_tailHead":flen_lpin_tailHead, "flen_rpin_tailHead":flen_rpin_tailHead,
                                                 "flen_withers_hipCon":flen_withers_hipCon, "flen_hipCon_tailHead":flen_hipCon_tailHead,
                                                 "flen_withers_cback":flen_withers_cback}


            # ******************************************************************************************#
            # *********************  FOR CHECKING DISTANCE FROM MASK EDGE ******************************#

            #if allVisible:
            #if all keypoints are not visible, the mask will be small as only a part of the cow is visible (Eg: in barn images).
            #when only parts of cow is visible, the interior keypoint could lie on the mask boundary. This distance is not what we are looking for.

            for kpName in keypoint_names:

                distance = None # you should return None value if keypoint is not visible!

                if kpName in visible and allVisible: # and maskContour is not None:
                    #pdb.set_trace()
                    # distFromMaskEdge = cv2.pointPolygonTest(maskContour, visible[kpName], measureDist = True) # Does not work

                    maskContourPoints = [x[0].tolist() for x in maskContour]
                    maskPolygon = Polygon(maskContourPoints)
                    poi = Point(visible[kpName]) #poi => point of interest

                    distance = maskPolygon.exterior.distance(poi) #exteerior => boundary/perimeter, not the exterior distance
                    #sign = -1 if  maskPolygon.contains(poi) else 1
                    #distance = sign*distance # distance = 0 if point is on the boundary

                if "distsFromMaskEdge" not in kpStatsDict:
                    kpStatsDict["distsFromMaskEdge"] = {kpName:distance}
                else:
                    kpStatsDict["distsFromMaskEdge"][kpName] = distance


            # ******************************************************************************************#
            # ***************************    PIN BONES AND HIP CONNECTOR *******************************#

            # check if angle and distance between pin bones and tail head at hip connector are within a margin
            ang_deviation, ang0, ang1 = self.get_ang_deviation(('tail_head', 'hip_connector', 'left_pin_bone'),
                                                               ('right_pin_bone', 'hip_connector', 'tail_head'),
                                                               visible=visible) #clockwise sweep rule

            len_deviation, len0, len1 = self.get_length_deviation(('hip_connector', 'left_pin_bone'),
                                                                  ('hip_connector', 'right_pin_bone'), visible=visible)

            kpStatsDict["pinBones_hcon"] = {"angles":{"angDev_pinBones_thead_at_hcon":ang_deviation, "ang_lpin_hcon_thead":ang0, "ang_rpin_hcon_thead":ang1}} #need to create a new dict entry
            kpStatsDict["pinBones_hcon"]["lengths"] = {"lenDev_pinBones_hcon":len_deviation, "len_lpin_hcon":len0, "len_rpin_hcon":len1}

            # also check if lpin-hcon-rpin angle is above a threshold (to eliminate cases where pin bones are detected at the same point)
            # ang_lpin_hc_rpin = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail! (See 5 June 2022 Journal log)
            _, ang0, _ = self.get_ang_deviation(('right_pin_bone', 'hip_connector', 'left_pin_bone'),
                                                ('right_pin_bone', 'hip_connector', 'left_pin_bone'), visible=visible) #clockwise sweep rule
            ang_lpin_hc_rpin = ang0
            kpStatsDict["pinBones_hcon"]["angles"]["ang_lpin_hcon_rpin"] = ang_lpin_hc_rpin

            # ******************************************************************************************#
            # ***************************    HIP BONES AND HIP CONNECTOR *******************************#

            # check if angle and distance between hip bones and tail head at hip connector are within a margin
            # this is the rigid part of the cow's body, so our constraints can be rigid
            ang_deviation, ang0, ang1 = self.get_ang_deviation(('tail_head', 'hip_connector', 'left_hip_bone'),
                                                               ('right_hip_bone', 'hip_connector', 'tail_head'),
                                                               visible=visible) #clockwise sweep rule
            len_deviation, len0, len1 = self.get_length_deviation(('hip_connector', 'left_hip_bone'),
                                                                  ('hip_connector', 'right_hip_bone'), visible=visible)

            kpStatsDict["hipBones_hcon"] = {"angles":{"angDev_hipBones_thead_at_hcon":ang_deviation, "ang_lhip_hcon_thead":ang0, "ang_rhip_hcon_thead":ang1}} #need to create a new dict entry
            kpStatsDict["hipBones_hcon"]["lengths"] = {"lenDev_hipBOnes_hcon":len_deviation, "len_lhip_hcon":len0, "len_rhip_hcon":len1}

            # also check if the left hip - hip_con - right hip angle > threshold (to prevent all 3 to be detected at the same point)
            # ang_lhip_hcon_rhip = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail
            _, ang0, _ = self.get_ang_deviation(('right_hip_bone', 'hip_connector', 'left_hip_bone'),
                                                ('right_hip_bone', 'hip_connector', 'left_hip_bone'), visible=visible) #clockwise sweep rule - it is ok here if angle is reflex, if you get poor results, change this to get_ang_deviation_old()
            kpStatsDict["hipBones_hcon"]["angles"]["ang_lhip_hcon_rhip"] = ang0

            # ******************************************************************************************#
            # ***************************    SHOULDERS AND WITHERS *******************************#

            # check if angle and distance between left and right shoulders with withers at center back are within a margin
            ang_deviation, ang0, ang1 = self.get_ang_deviation(('left_shoulder', 'center_back', 'withers'),
                                                               ('withers', 'center_back', 'right_shoulder'), visible=visible) #order of keypoints matter - clockwise sweep rule!
            _, ang2, _ = self.get_ang_deviation(('right_shoulder', 'withers', 'left_shoulder'),
                                                               ('right_shoulder', 'withers', 'left_shoulder'), visible=visible) #to avoid neck being detected as withers #clockwise sweep rule
            len_deviation, len0, len1 = self.get_length_deviation(('center_back', 'left_shoulder'),
                                                                  ('center_back', 'right_shoulder'), visible=visible)

            kpStatsDict["shldrs_withers"] = {"angles":{"angDev_shldrs_withers_at_cback":ang_deviation, "ang_lshldr_cback_withers":ang0, "ang_rshldr_cback_withers":ang1, "ang_lshldr_withers_rshldr":ang2}} #need to create a new dict entry
            kpStatsDict["shldrs_withers"]["lengths"] = {"lenDev_shldrs_cback":len_deviation, "len_lshldr_cback":len0, "len_rshldr_cback":len1}

            # also check if the lshoulder - center back - rshoulder angle > threshold (to eliminate cases where both shoulders and wihters to be detected at the same point)
            # ang_lshld_cback_rshld = min(abs(ang0) + abs(ang1), 360 - (abs(ang0)+abs(ang1))) - Fail - adding two angles with abs will give positive result even if shouder points are detected at the same location
            _, ang0, _ = self.get_ang_deviation(('left_shoulder', 'center_back', 'right_shoulder'),
                                                ('left_shoulder', 'center_back', 'right_shoulder'),
                                                visible=visible)  # get the angle directly - again - double computation, I know - cut me some slack #clockwise sweep rule
            kpStatsDict["shldrs_withers"]["angles"]["ang_lshldr_cback_rshldr"] = ang0


            # ******************************************************************************************#
            # ***************************    SHOULDERS AND PIN BONES *******************************#

            # check if shoulders and pin bones are not detected on the same side of the cow
            # to do this, check if angle made by left shoulder and left pin bone at center back is > 90 derees, do the same for right shoulder and right pin bone
            _, ang0, ang1 = self.get_ang_deviation(('left_pin_bone', 'center_back', 'left_shoulder'),
                                                   ('right_shoulder', 'center_back', 'right_pin_bone'), visible=visible) #clockwise sweep rule

            kpStatsDict["shldrs_pinBones"] = {"angles":{"ang_lshldr_cback_lpin":ang0, "ang_rshldr_cback_rpin":ang1}} #need to create a new dict entry

            # ******************************************************************************************#
            # *****************************   CENTER-BACK AND HIP BONES  *******************************#
            # adding this extra check to help predicte misplaced keypoint better
            ang_deviation, ang0, ang1 = self.get_ang_deviation(('hip_connector', 'center_back', 'left_hip_bone'),
                                                   ('right_hip_bone', 'center_back', 'hip_connector'), visible=visible)  #clockwise sweep rule

            kpStatsDict["hipBones_cback"] = {"angles": {"angDev_hipBones_hcon_at_thead": ang_deviation,
                                                         "ang_lhip_cback_hcon": ang0,
                                                         "ang_rhip_cback_hcon": ang1}}

            # ******************************************************************************************#
            # *****************************   HIP BONES AND SHOULDERS *******************************#
            # adding this extra check to help predicte misplaced keypoint better
            ang_deviation, ang0, ang1 = self.get_ang_deviation(('withers', 'left_shoulder', 'left_hip_bone'),
                                                               ('right_hip_bone', 'right_shoulder', 'withers'),
                                                               visible=visible) #clockwise sweep rule
            ang_deviation2, ang2, ang3 = self.get_ang_deviation(('left_hip_bone', 'center_back', 'left_shoulder'),
                                                                ('right_shoulder', 'center_back', 'right_hip_bone'),
                                                                visible=visible) #clockwise sweep rule

            ## to avoid both pin bones and the tail head being detected the same location
            #_, ang4, ang5 = self.get_ang_deviation(('left_shoulder', 'left_hip_bone', 'left_pin_bone'),
            #                                       ('right_pin_bone', 'right_hip_bone', 'right_shoulder'),
            #                                       visible=visible) # CHECK FOR CLOCKWISE SWEEP RULE IF YOU WANT TO INCLUDE THIS

            kpStatsDict["hipBones_shldrs"] = {"angles": {"angDev_hipBones_withers_at_shldrs": ang_deviation,
                                                        "ang_lhip_lshldr_withers": ang0, "ang_rhip_rshldr_withers": ang1,
                                                                 "angDev_hipBones_shldrs_at_cback":ang_deviation2,
                                                       "ang_lhip_cback_lshldr":ang2, "ang_rhip_cback_rshldr":ang3}}

            #kpStatsDict["hipBones_shldrs"]["angles"]["ang_lpin_lhip_lshldr"] = ang4
            #kpStatsDict["hipBones_shldrs"]["angles"]["ang_rpin_rhip_rshldr"] = ang5

            # ******************************************************************************************#
            # *******************************   HIP BONES AND PIN BONES  *******************************#

            # check if pin bones and hip bones are not detected close to each other
            # do this by checking if the angle between pin bones and hip bones at the hip connector is > than a threshold
            _, ang0, ang1 = self.get_ang_deviation(('left_pin_bone', 'hip_connector', 'left_hip_bone'),
                                                   ('right_hip_bone', 'hip_connector', 'right_pin_bone'), visible=visible) #clockwise sweep rule

            # to avoid both hip bones/ both pin bones being detected at the same location
            _, ang2, ang3 = self.get_ang_deviation(('left_hip_bone', 'left_pin_bone', 'tail_head'),
                                                   ('tail_head', 'right_pin_bone', 'right_hip_bone'),
                                                   visible=visible) #clockwise sweep rule

            #for some redundancy
            len_deviation, len0, len1 = self.get_length_deviation(('left_hip_bone', 'left_pin_bone'), ('right_hip_bone', 'right_pin_bone'), visible=visible)

            kpStatsDict["hipBones_pinBones"] = {"angles":{"ang_lhip_hcon_lpin": ang0, "ang_rhip_hcon_rpin": ang1, "ang_lhip_lpin_thead":ang2, "ang_rhip_rpin_thead":ang3}} #need to create a new dict entry
            kpStatsDict["hipBones_pinBones"]["lengths"] = {"lenDev_hips_pins":len_deviation, "len_lhip_lpin":len0, "len_rhip_rpin":len1}

            # ******************************************************************************************#
            # *******************************    TAIL-HEAD AND WITHERS   *******************************#
            # **********************************     SPINE POINTS      *********************************#

            # check if tail head and withers are not detected at the same point (or anywhere very close to each other)
            # We do this in 3 steps,
            #   - check if angle between center back and tail head at hip connector is above a certain angle
            #   - check if angle between withers and hip connector at center back is above a certain angle
            #   - check if angle between withers and tail head at hip connector is above a certain angle

            # This could also be checked by fitting a 3rd order polynomial on the spine points and checking if the coeff of x^3 ~=0 and coeff of x^2 is small. (The bend should be within a threshold)
            # We shall measure this and keep it as option2. We shall switch to this as the primary option if it works.

            _, ang0, ang1 = self.get_ang_deviation(('center_back', 'hip_connector', 'tail_head'),
                                                   ('withers', 'center_back', 'hip_connector'), visible=visible)
            #clockwise sweep rule does not matter - even if cow is bent only in one direction during training, remember that we also measure angles with LR flipped image. That makes sure we get the right upper and lower angle limits.

            _, ang2, _ = self.get_ang_deviation(('withers', 'hip_connector', 'tail_head'),
                                                ('withers', 'hip_connector', 'tail_head'), visible=visible)
            # clockwise sweep rule does not matter - even if cow is bent only in one direction during training, remember that we also measure angles with LR flipped image. That makes sure we get the right upper and lower angle limits.

            kpStatsDict["spinePoints"] = {"angles":{"ang_cback_hcon_thead":ang0, "ang_witr_cback_hcon":ang1, "ang_witr_hcon_thead":ang2}} #need to create a new dict entry

            # NOTE THAT COEFF_x0 GIVES OFFSET, COEFF_x1 GIVES SLOPE - WHICH ARE NOT REQUIRED - COW AN BE ANYWHERE AND ROTATED AT ANY ANGLE, WE JUST NEED DEGREE OF BEND, but these coeffs do not directly translate to angle and bend!
            # TRY TO PUT IT IN POLAR COORDINATES WITH r, theta and higher orders of curvature to use it.
            # on doing that, we can ignore r and theta and concentrate only on the higher order bends.
            # Turining this feature off for now!
            #curve = None
            #degree = 3
            #if 'tail_head' in visible and 'hip_connector' in visible and 'center_back' in visible and 'withers' in visible:
            #    x1, y1 = visible['tail_head']
            #    x2, y2 = visible['hip_connector']
            #    x3, y3 = visible['center_back']
            #    x4, y4 = visible['withers']
            #
            #    n = np.array([1, 1, 1, 1])
            #    curve = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], w=np.sqrt(n), deg=degree)  # ax^3 + bx^2 + cx + d #we choose degree 3 instead of 3 to allow for two bends. We might get two bends if there is an error
            #    #check if this works, else, increase degree and check
            #
            ##kpStatsDict["spinePoints"]["curve"] = curve
            #kpStatsDict["spinePoints"]["curve"] = {}
            #for i in range(degree+1): #save each coeff separately, helps to get min, max stats later
            #    kpStatsDict["spinePoints"]["curve"][f"coeff_x{degree-i}"] = curve[i] if curve is not None else None #len(curve) = degree

            return kpStatsDict
        else:
            return None #this happens by default, but I am writing this here to remind myself to handel None returns.

    def flattenDict(self, dict1, level=0):  # modified by Manu. https://stackoverflow.com/questions/5938125/converting-a-nested-dictionary-to-a-list
        '''
        Flattens a dictionary into a list
        :param dict1: input dictionary
        :param level: the level in the nested dictionary - used to mark the level in the list after flattening - helps in reconstruction.
        :return:
        '''
        result = []
        for key, val in dict1.items():
            keyInList = f"{level}~{key}"  # "~"*level+key
            result.append(keyInList)
            if type(val) is dict:  # dict here is a keyword
                result.extend(self.flattenDict(val, level=level + 1))
            else:
                result.append(val)
        return result

    def buildNestedDict(self, inList):
        '''
        Used to generate the nested dictionary from the flattened list
        This is done after generating the dict with the min, max, mean or other stats
        :param inList: the flattend list
        :return: the nested dictionary
        '''
        result = {}

        tempList = inList.copy()
        outList = ['~start']  # '~start' is a seed key
        while (len(tempList) > 1):

            #print(f"tempList = {tempList}")
            #print(f"Len of tempList = {len(tempList)}")

            for idx, el in enumerate(tempList):
                #print(f"*****Entering For Loop**********")

                if type(el) is not str or (type(el) is str and '~' not in el):  # element is a value
                    #print(f"Current element is value, current element = {el}")

                    # YAML files create issues in writing numpy objects to files - it doesn't save it in human readable format!
                    # So, we change to builtin type. # https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
                    if type(el).__module__ == 'numpy':
                        print(f"\n\nElement is of numpy type. Changing to builtin type.")
                        el = el.item()
                        #pdb.set_trace()

                    if idx == 0:
                        # first element itself is a value, this happens when first element itself is a dictionary
                        # this occurs in the due course of the process
                        # we cannot check the prev element in this case and python gives out of bounds error
                        # this element also contains the main key
                        # so, we append it to the outlist and continue.
                        outList.append(el)
                        #print(f"*****First element is value*******")
                        continue

                    # prev_el = tempList[idx-1]
                    prev_el = outList[-1]
                    if type(prev_el) is str and '~' in prev_el:  # prev element is a key
                        #print(f"prev el is key, prev element = {prev_el}")
                        temp = {prev_el.split('~')[-1]: el}  # removing <level_number>~ from the key's name. Once we put it inside the dictionary as a key, there is no need to retain the ~ as we will not be checking if it is a key or not again

                        outList.pop()
                        outList.append(temp)
                    elif type(prev_el) is dict and type(el) is dict:  # else: #prev element is already a dictionary
                        #print(f"Prev element is dict, prev el = {prev_el}; current element is dict, current el = {el}")
                        prev_el_key = list(prev_el)[0]
                        el_key = list(el)[0]
                        prev_el[prev_el_key][el_key] = el[el_key]  # inserting current dict into prev dict
                        outList.pop()
                        outList.append(prev_el)
                else:  # element is a key
                    #print(f"Current element is key, current el = {el}")
                    outList.append(el)

            #print(f"\ntempList again = {tempList}")
            #print(f"\n\nOutlist = {outList}")
            #print(f"\nLen of outlist = {len(outList)}")

            # print(f"Type of tempList = {type(tempList)}, type of outList = {type(outList)}")

            tempList = []
            tempList = outList
            outList = []

            #print(f"Len of tempList again = {len(tempList)}")
            # pdb.set_trace()
            if len(tempList) == 1:
                break
        return tempList[0]['start']

    def flipLR_Keypoints(self, keypoints_array, keypoint_names, frameW=1920, frameH=1080):
        '''
        Reflects the image across Y axis (Mirror effect).
        This function could be used to make stats independent of cow's orientation.
        Otherwise, if cows are seen only from one side in the training set, the stats might show limits that are biased to just one side,
        and that would affect rule checks.

        Steps:
        1. They keypoints locations are flipped.
        2. Left and right keypoints are interchanged.
        :param keypoints_array:
        :return: flippedKPs_array
        '''

        kpn = keypoint_names
        flippedKPs_array = keypoints_array.copy()

        # Step 1
        for idx, value in enumerate(keypoints_array):
            x, y, visibility = value
            #flippedKPs_array[idx] = [frameW-1-x, frameH-1-y, visibility] #this is flipping across both x and y axes -> Totally wrong!
            flippedKPs_array[idx] = [frameW-1-x, y, visibility]

        # Step 2
        temp =  flippedKPs_array[kpn.index('left_shoulder')].copy()
        flippedKPs_array[kpn.index('left_shoulder')] = flippedKPs_array[kpn.index('right_shoulder')].copy()
        flippedKPs_array[kpn.index('right_shoulder')] = temp.copy()

        temp = flippedKPs_array[kpn.index('left_hip_bone')].copy()
        flippedKPs_array[kpn.index('left_hip_bone')] = flippedKPs_array[kpn.index('right_hip_bone')].copy()
        flippedKPs_array[kpn.index('right_hip_bone')] = temp.copy()

        temp = flippedKPs_array[kpn.index('left_pin_bone')].copy()
        flippedKPs_array[kpn.index('left_pin_bone')] = flippedKPs_array[kpn.index('right_pin_bone')].copy()
        flippedKPs_array[kpn.index('right_pin_bone')] = temp.copy()

        return flippedKPs_array

    def computeKPRuleStats_onDataset(self, optimizeForKpRules=True, outRootDir='./'):
        '''
        Computes stats of keypoints in the dataset.
        These could be used to set the thresholds for cow keypoints
        :param optimizeForKpRules: If True, will return values optimized for keypoint rule checker limits. Turn it to False to get measurements from actual data.
                                    -   Measure stats also with lef-right flipped keypoints, to make measurement limits independent of cow orientation.
                                    -   Will  set all minimum deviation values to zero.
        :param outRootDir: The root directory where the stats will be saved.
        :return: None
        
        refer: create_KP_cattlog.py for handling coco datasets.
        
        '''
        pass

        cats = self.coco.loadCats(self.coco.getCatIds())
        catIds = self.coco.getCatIds(catNms=[cats[0]]) #does not work
        cat_ids = self.coco.getCatIds() #works
        imgIds = self.coco.getImgIds(catIds=catIds)

        print(f"\nThere are a total of {len(imgIds)} annotated images")
        print(f"CatIds = {catIds}, cats = {cats}, \nimgIds = {imgIds}")

        keypoint_names = kpn = cats[0]['keypoints']
        # kp connection rules is also in cats[0]['skeleton'] but it has no color info
        keypoint_connection_rules = keypoint_connection_rules = [(kpn[1 - 1], kpn[4 - 1], (0, 0, 255)), (kpn[1 - 1], kpn[5 - 1], (0, 255, 0)), (kpn[1 - 1], kpn[2 - 1], (255, 0, 0)), (kpn[2 - 1], kpn[3 - 1], (255, 255, 0)), (kpn[2 - 1], kpn[4 - 1], (0, 255, 255)), (kpn[3 - 1], kpn[4 - 1], (0, 0, 0)), (kpn[3 - 1], kpn[7 - 1], (255, 255, 255)), (kpn[4 - 1], kpn[6 - 1], (255, 128, 128)), (kpn[4 - 1], kpn[5 - 1], (128, 255, 128)), (kpn[4 - 1], kpn[7 - 1], (128, 128, 255)), (kpn[5 - 1], kpn[6 - 1], (255, 255, 128)), (kpn[5 - 1], kpn[8 - 1], (255, 128, 255)), (kpn[6 - 1], kpn[7 - 1], (128, 255, 255)), (kpn[6 - 1], kpn[8 - 1], (255, 128, 64)), (kpn[6 - 1], kpn[10 - 1], (255, 64, 128)), (kpn[6 - 1], kpn[9 - 1], (128, 255, 64)), (kpn[7 - 1], kpn[10 - 1], (128, 64, 255)), (kpn[8 - 1], kpn[9 - 1], (64, 255, 128)), (kpn[9 - 1], kpn[10 - 1], (64, 128, 255))]

        list_of_kpStatsLists = []

        for imgId in imgIds:

            #imgPath = imgDir + '/' + self.coco.imgs[imgId]['path'].split('/')[-1]
            imgName = self.coco.imgs[imgId]['path'].split('/')[-1]
            print(f"Currently processing image: {imgName}")

            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=cat_ids[0], iscrowd=None)
            annos = self.coco.loadAnns(annIds)  # there is only one annotation
            #print(f"annotations of the given category = {annos}")

            #mask = coco.annToMask(annos[0])  # bianry mask with values in [0,1], len of annos is just 1 - there is only one annotated cow per image

            for anno in annos:
                mask = self.coco.annToMask(anno)  # bianry mask with values in [0,1] #mask is needed to compute cowW and cowH (length)
                mask = mask.astype(np.uint8) * 255

                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                if len(contours) > 1:
                    print(f"More than one contour detected. Not Breaking. Using largest contour.")
                    contours = sorted(contours, key=cv2.contourArea)[::-1]  # sort in descending order

                    # cv2.imwrite(f"multi_contour_image_{frameCount}.jpg", overlap)
                    # break
                maskContour = contours[0]
                cowW, cowH = self.get_cowW_cowH(mask_img=mask)

                if maskContour is None:
                    print(f"Mask contour is none. Please provied mask input.")

                keypoints = np.array(anno['keypoints'], int).reshape((-1,3))
                print(f"Keypoints = {keypoints}")
                kpStatsDict =  self.computeKPRuleStats_perInstance(keypoints=keypoints, cowW=cowW, cowH=cowH, keypoint_connection_rules=keypoint_connection_rules, keypoint_names=kpn, keypoint_threshold=0.5, allKPs_mustBeVisible=False, maskContour=maskContour, frameW=1920, frameH=1080)
                kpStatsList = self.flattenDict(kpStatsDict) #flattened list
                print(f"Flattend KP Stats dict = \n{kpStatsList}")

                #nestedDict = self.buildNestedDict(kpStatsList) #build it after computing stats
                #print(f"\nRebuilt nested dict = {nestedDict}")

                list_of_kpStatsLists.append(kpStatsList)

                if optimizeForKpRules:
                    flippedKPs = self.flipLR_Keypoints(keypoints_array=keypoints, keypoint_names=kpn, frameW=1920, frameH=1080)

                    #shold also flip mask contours
                    flippedMask = cv2.flip(mask, flipCode=1) # flipCode = 1 => flip horizontally, 0=> vertically, -1 => both vertically and horizontally
                    #contours of flipped mask != flipped contours. The contours are recalculated for the flipped mask. They could be slightly different from directly flipping the contours.
                    # I did not flip the contours directly as it would make me code a bit more (I was lazy). And, it should not matter as we would be making our system robust to approximation errors.
                    contours2, hierarchy2 = cv2.findContours(flippedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    if len(contours2) > 1:
                        print(f"More than one flipped contour detected. Not Breaking. Using largest flipped contour.")
                        contours2 = sorted(contours2, key=cv2.contourArea)[::-1]  # sort in descending order

                    contour_ofFlippedMask = contours2[0]

                    print(f"Flipped Keypoints = {flippedKPs}")
                    kpStatsDict = self.computeKPRuleStats_perInstance(keypoints=flippedKPs, cowW=cowW, cowH=cowH,
                                                                      keypoint_connection_rules=keypoint_connection_rules,
                                                                      keypoint_names=kpn, keypoint_threshold=0.5,
                                                                      allKPs_mustBeVisible=False, maskContour=contour_ofFlippedMask,
                                                                      frameW=1920, frameH=1080)
                    kpStatsList = self.flattenDict(kpStatsDict)  # flattened list
                    #print(f"Flattend KP Stats dict = \n{kpStatsList}")
                    # nestedDict = self.buildNestedDict(kpStatsList) #build it after computing stats
                    # print(f"\nRebuilt nested dict = {nestedDict}")

                    list_of_kpStatsLists.append(kpStatsList)


        outDirPath = f"{outRootDir}/statOutputs/"
        os.makedirs(outDirPath, exist_ok=True)
        
        pd.DataFrame(np.array(list_of_kpStatsLists)).to_csv(f"{outDirPath}/kp_statsLists.csv")

        #now that we have the list of all the stats for all the cow instances, we shall proceed to calculate the min and max
        kpStatsArray = np.array(list_of_kpStatsLists)

        maxStatList = []; minStatList = []; meanStatList = []

        for idx, element in enumerate(kpStatsArray[0]): #iterate across columns - choosing row 0, you could choose any row
            if type(element) is str and '~' in element: #element is a key
                minStatList.append(element)
                maxStatList.append(element)
                #meanStatList.append(element)
            else:
                #pdb.set_trace()
                column = kpStatsArray[:, idx]
                minStatList.append(min(column[column != np.array(None)]))
                maxStatList.append(max(column[column != np.array(None)]))
                #meanStatList.append(np.mean(column[column != np.array(None)]))

        print(f"Min array = {minStatList}")
        print(f"Max array = {maxStatList}")
        #print(f"Mean array = {meanStatList}")
        
        minStatDict = self.buildNestedDict(minStatList)
        maxStatDict = self.buildNestedDict(maxStatList)

        print(f"Min stat dict = {minStatDict}")
        print(f"Max stat dict = {maxStatDict}")

        # Setting all deviation min values to 0 (lenDev and angDev).
        # They are usually a little above 0. But test examples can have 0 deviations.
        # So, we set min allowed deviation to the ideal 0 value.
        if optimizeForKpRules:
            minStatNDict = NestedDict(minStatDict)
            for rule, stats in minStatNDict.items():
                if "Dev_" in rule[-1]:
                    minStatNDict[rule] = 0
            minStatDict = minStatNDict.to_dict()
            print(f"Min stat dict after optimizing for KP rules =\n{minStatDict}")

        datasetKpStatsDict = {"minStatDict":minStatDict, "maxStatDict":maxStatDict} #the overall dictionary

        #saving to YAML file to improve human readability and for ease of editing
        yaml.dump(datasetKpStatsDict, open(f"{outDirPath}/datasetKpStatsDict_{self.dsetName}.yml", "w")) #this is to be used for setting upper and lower bounds for kp rule checks
        pickle.dump(datasetKpStatsDict, open(f'{outDirPath}/datasetKpStatsDict_{self.dsetName}.p', 'wb')) #this is to be used for setting upper and lower bounds for kp rule checks

        return datasetKpStatsDict


    def get_length_deviation(self, pts0, pts1, visible):
        '''

        :param pts0: tuple with names of the two keypoints forming the line segment
        :param pts1: tuple with names of the two keypoints forming the line segment
        :param visible: the visible keypoints dictionary.
        :return: (deviation, len0, len1)
        '''

        pt00, pt01 = pts0
        pt10, pt11 = pts1

        #check for visibility - as allKPs_mustBeVisible condition could be off
        len_deviation = None; l1 = None;  l2 = None
        if pt00 in visible and pt01 in visible:
            vec0 = np.array(visible[pt01]) - np.array(visible[pt00])
            l1 = np.linalg.norm(vec0).item() # saving as builtin data type as numpy types are not saved properly in YAML files
        if pt10 in visible and pt11 in visible:
            vec1 = np.array(visible[pt11]) - np.array(visible[pt10])
            l2 = np.linalg.norm(vec1).item() # saving as builtin data type as numpy types are not saved properly in YAML files
        if l1 is not None and l2 is not None:
            len_deviation = (np.abs(l1 - l2) / (max(l1, l2) + 1e-10)).item() # saving as builtin data type as numpy types are not saved properly in YAML files

        #remove later
        #logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> l1={l1}, l2={l2}, len_deviation = {len_deviation}")
        #print(f"l1={l1}, l2={l2}, len_deviation = {len_deviation}")

        return len_deviation, l1, l2 #returns Nones in place of missing items

    def get_ang_deviation(self, pts0, pts1, visible):
        '''
        Computes signed angles.
        NOTE: THIS COMPUTATION FOLLOWS LEFT HAND RULE - WITH THUMB POINTING OUT OF SCREEN (CLOCKWISE SWEEP). YOU NEED TO SUPPLY VECTORS IN THE LEFT HAND ORDER FOR THIS TO WORK CORRECTLY.
        Example:
        pts0 = [thead, hcon, lpin] => the first vector v00 - hcon-thead will sweep clockwise to the second vector hcon-lpin - to generate an acute angle.
        pts1 = [rpin, hcon, thead] => the first vector v01 - hcon-rpin will sweep clockwise to the second vector hcon-thead - to generate an acute angle.
        If pts1 = [thead, hcon, rpin], then the first vector hcon-thead will sweep clockwise to hcon-rpin generating a reflex angle (>180 degrees),
        this would cause the angle deviation to be very large - and make the rule checker limits also very large!

        Note2: The whole thing should also work if anti-clockwise sweep convention is followed throughout. This would just replace all acute angles with their reflex equivalents (360-angle) and the min and max stat dicts will represent the opposite limits.
        It is just that we should stick to one convention throughout and not mix and match. I have chosen to go with the clockwise sweep convention.

        visible: the visible keypoints dictionary
        visible: the visible keypoints dictionary

        pts1 and pts2 are tuples with names of 3 keypoints forming the angle.
        angle is formed at second keypoint (the centre one)
        :return: (deviation, angle0, angle1)
        '''
        pt00, pt01, pt02 = pts0
        pt10, pt11, pt12 = pts1

        # check for visibility - as allKPs_mustBeVisible condition could be off
        ang_deviation = ang0 = ang1 = None

        if pt00 in visible and pt01 in visible and pt02 in visible:
            vec00 = np.array(visible[pt00]) - np.array(visible[pt01]);
            vec01 = np.array(visible[pt02]) - np.array(visible[pt01]);
            ang00 = np.degrees(np.arctan2(vec00[1], vec00[0])) # angle of vec00 with +X axis
            ang01 = np.degrees(np.arctan2(vec01[1], vec01[0])) # angle of vec01 with +X axis
            ang0 = ang01 - ang00 if (ang01-ang00) >=0 else (ang01-ang00)+360

            ang0 = ang0.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        if pt10 in visible and pt11 in visible and pt12 in visible:
            vec10 = np.array(visible[pt10]) - np.array(visible[pt11]);
            vec11 = np.array(visible[pt12]) - np.array(visible[pt11]);
            ang10 = np.degrees(np.arctan2(vec10[1], vec10[0]))  # angle of vec10 with +X axis
            ang11 = np.degrees(np.arctan2(vec11[1], vec11[0]))  # angle of vec11 with +X axis
            ang1 = ang11 - ang10 if (ang11 - ang10) >= 0 else (ang11 - ang10)+360

            ang1 = ang1.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        if ang0 is not None and ang1 is not None:
            ang_deviation = np.abs(ang0 - ang1) / (max(ang0, ang1) + 1e-10)
            ang_deviation = ang_deviation.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        return (ang_deviation, ang0, ang1) #returns Nones in place of missing items


    def get_ang_deviation_old(self, pts0, pts1, visible):
        '''
        Does not differentiate between (0 to 180) and (0 to -180) degrees

        visible: the visible keypoints dictionary

        pts1 and pts2 are tuples with names of 3 keypoints forming the angle.
        angle is formed at second keypoint (the centre one)
        :return: (deviation, angle0, angle1)
        '''
        pt00, pt01, pt02 = pts0
        pt10, pt11, pt12 = pts1

        # check for visibility - as allKPs_mustBeVisible condition could be off
        ang_deviation = ang0 = ang1 = None

        if pt00 in visible and pt01 in visible and pt02 in visible:
            vec00 = np.array(visible[pt00]) - np.array(visible[pt01]);
            vec01 = np.array(visible[pt02]) - np.array(visible[pt01]);
            ang0 = np.arccos(np.dot(vec00, vec01) / (np.linalg.norm(vec00) * np.linalg.norm(vec01))) / np.pi * 180
            #ang0 = min(abs(ang0), 360 - abs(ang0)) # to always return the smaller (inner angle), positive angle #uncomment if necessary
            ang0 = ang0.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        if pt10 in visible and pt11 in visible and pt12 in visible:
            vec10 = np.array(visible[pt10]) - np.array(visible[pt11]);
            vec11 = np.array(visible[pt12]) - np.array(visible[pt11]);
            ang1 = np.arccos(np.dot(vec10, vec11) / (np.linalg.norm(vec10) * np.linalg.norm(vec11))) / np.pi * 180
            #ang1 = min(abs(ang1), 360 - abs(ang1)) # to always return the smaller (inner angle), positive angle #uncomment if necessary
            ang1 = ang1.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        if ang0 is not None and ang1 is not None:
            ang_deviation = np.abs(ang0 - ang1) / max(ang0, ang1)
            ang_deviation = ang_deviation.item() # saving as builtin data type as numpy types are not saved properly in YAML files

        return (ang_deviation, ang0, ang1) #returns Nones in place of missing items

    def compute_bitVectorDissimilarity(self, X):
        '''
        Computes the dissimiilarity between sets of bit vectors in number of dissimilar bits.

        :param X: Array - m features X n bit vectors
        :return: array - n bit vectors X n bit vectors (something like a correlation matrix). Each value in this array will contain the number of bits by which the row bit vector differs with the column bit vector in the input matrix.
        '''

        A = 2*X - 1 #takes 1,0 matrix to 1,-1 - scaled and shifted X
        intr = np.matmul(A.T,A) #intermediate result
        m, n = A.shape

        '''
        By Manu:
                
        Calculating number of dissimilar bits is equivalent to taking the bitwise XOR results of the two vectors and summing the result.    
        This however cannot be directly, quickly and efficiently computed for a bunch of bit vectors. 
        If only there existed a technique where we could do A.T * A where * is similar to matrix multiplication but does XOR in place of multiplication.
        To overcome this, here is an elaborate scheme.
        
        Steps:
        1. First create a polarized bit vector - polarized vector has -1s in place of 0s of the bit vector.  
        2. Stack these together as a matrix X of shape m features X n bit vectors.
        3. Obtain the cross correaltion matrix X.T x X.
        4. Recover the number of dissimilar bits from the values in the cross correlation matrix using the following formula.
        
        Each bit vector has m bits. m = lenght of bit vector
        When comparing bit vectors of two cows
        let k = number of similar bits
        => m-k = number of dissimilar bits
        let r = intermediate result = result obtained by taking dot product of 1,-1 vectors of the two cows
        clearly,
        r = 1*k + (-1) (m-k)
        r = k -m + k
        r = 2k -m  --(1)
        OR
        k = (m+r)/2 --(2)
        
        Given bit vectors, we know m. We can compute r as descibed above.
        We can compute k using (2)
        
        But we need the number of dissimilar bits (m-k)
        m-k = m - ((m+r)/2) = (2m-m-r)/2 = (m-r)/2
        number of dissimilar bits = (m-k) = (m-r)/2
        
        '''
        disMat = ((m-intr)/2).astype(int) #the dissimilarity matrix. Measures dissimilarity in bits of dissimilarity. This is a symmetric matrix.
        #print(f"Dissimilarity matrix = \n{disMat}")

        return disMat

    def compute_dataset_disMat(self, cowDataMatDict):
        '''
        Computes the dissimilarity matrix for all cows in the dataset.

        :param cowDataMatDict: the cow data matrix dictionary stored in the pickle file. This has all cows in the dataset.
                This dictionary is of the form {'<cowID>':{'blk128':'<bit_vector>', 'blk64':<bit_vector>',.....,'blk<blkSize>':'<bit_vector>'}}
                The smallest blk size as of today is 16x16 pixels => blk16 is the last key for a cow id.
                The cow ids are 4 digit numbers. CowIDs and bit vectors are stored as strings.

                to obtain dictionary from pickle file,
                dict = pickle.load(open('<path_to_file>','rb'))

        :return: the dissimilarity matrix of the entire dataset.
        '''

        nCows = len(cowDataMatDict.keys())

        blkSizes = list(list(cowDataMatDict.items())[0][1].keys())
        #blkSizes = ['blk128'] #['blk16'] #check with other block sizes later

        for blkSize in blkSizes:
            X = [] # the bit vector matrix

            #aside
            bv2cowID_dict = {} #bit vector to cow id dictionary for knowing which 2 cows have the same bvs
            nBVs_moreThanOneCow = 0 #number of bit vectors with more than one cow
            BVs_moreThanOneCow = []

            for cowID, val in cowDataMatDict.items():
                bvStr = val[blkSize] #bit vector as a string
                bitVec = [int(x) for x in bvStr]
                #print(f"{cowID}:{bitVec}")
                m = len(bitVec)
                X.append(bitVec)

                #aside
                if bvStr not in bv2cowID_dict:
                    bv2cowID_dict[bvStr] = [cowID]
                else:
                    if len(bv2cowID_dict[bvStr]) == 1:
                        nBVs_moreThanOneCow += 1
                        BVs_moreThanOneCow.append(bvStr)
                    bv2cowID_dict[bvStr].append(cowID)

            X = np.array(X).T
            #print(f"Shape of bit vector matrix X = {X.shape}") # m features x n cows

            #hardcoding for testing - remove later
            #X = X[:,135:150]
            #or
            #b1 = [1, 0, 1, 1]; b2 = [1, 1, 0, 1]; b3 = [1, 0, 0, 1]; b4 = b5 = b6 = b3
            #X = np.array([b1,b2,b3,b4,b5,b6], int).T
            #m, nCows = X.shape

            disMat = self.compute_bitVectorDissimilarity(X) #this is a symmetric matrix
            print(f"The cows are dissimilar in these number of bits.")
            print(f"Dissimilarity matrix for {blkSize}:\n{disMat}")
            print(f"DisMat shape = {disMat.shape}\n")
            pd.DataFrame(disMat).to_csv(f"./output/statOutputs/disMat_{blkSize}.csv")

            #taking just the upper triangluar matrix to avoid repeats,
            upTriIndices = np.triu_indices(nCows, 1) #calculates upper triangular indices of nCows X nCows matrix. 1 indicates that we start from 1 diagonal to the right of the main diagonal. This ensures we ignore bit vector dissimilarities of 0 when both bit vectors are from the same cow.
            disMat_noReps = disMat[upTriIndices] #dissimilarity values without repeats

            #hist= np.histogram(disMat_noReps, bins=np.arange(0,m+1))
            hist = np.bincount(disMat_noReps)
            print(f"Histogram of dissimilarity at {blkSize}:(Note: We ignore 0 dissimilarites of same cow matches and also ignore duplicate counts from symmetric disMat)")
            print(f"Nbits :{np.arange(len(hist))}\nCounts:{hist}")
            pd.DataFrame(np.vstack([np.arange(len(hist)), hist])).to_csv(f"./output/statOutputs/bincountHist_{blkSize}.csv")

            #THIS IDEA DOES NOT WORK **************
            #now we need to count the number of cows that have the same bit vectors at this block size
            #a 0 in the dissimilarity matrix indicates that the row and the col cows have same bit vectors.
            #dissimilarity matrix is of size nCows X nCows.
            #the upper tri matrix we selected has size (nCows x nCows)/2 - nCows  -> (-nCows as we have removed the main diagonal)
            #the total number of 0s in the disMat gives the value of nC2 -> the total number of cow combinations, 2 cows at a time, which has 0 bit difference
            #let v = nC2. Then, n gives the total number of cows having at least one other cow that has the same bit vector as itself.
            # v = nC2 = n!/((n-2)!2!) = n(n-1)/2   --> n^2 -n -2v = 0.
            #solving for n in the above quadratic equations will give us the required result.
            #THIS WORKS ONLY WHEN BIT VECTORS OF ALL COWS THAT ARE SIMILAR ARE THE SAME. THIS DOES NOT WORK IF THERE ARE MORE THAN ONE BIT-VECTORS RESULTING IN MORE THAN ONE CLUSTER OF SIMILAR COWS!

            print(f"Number of bit vectors with more than one cow at {blkSize} = {nBVs_moreThanOneCow}\n")
            #print(f"BV2cowID Dict = {bv2cowID_dict}\n\n")
            print(f"BitVector clusters with more than one cow at {blkSize}:\n")
            for idx, bv in enumerate(BVs_moreThanOneCow):
                print(f"Cluster {idx}: {bv2cowID_dict[bv]}")
            print() #for new line



if __name__ == "__main__":
    #dstat = DatasetStats(dsetFilePath="../../Data/CowTopView/Datasets/kp_dataset_v4/annotations/kp_dataset_v4_train.json", img_dir="")

    #ds1 = DatasetStats(dsetFilePath="../../Data/CowTopView/Datasets/kp_dataset_v4/annotations/kp_dataset_v4_train.json", img_dir="../../Data/CowTopView/Datasets/kp_dataset_v4/images/images_train/")
    ds1 = DatasetStats(dsetFilePath="../../Data/CowTopView/Datasets/kp_dataset_v4/annotations/kp_dataset_v4_train.json")

    #b1 = [1,0,1,1]; b2 = [1,1,0,1]; b3 = [1,0,0,1]
    #B = np.array([b1,b2,b3], int).T
    #print(f"Entered matrix B = \n{B}\n")
    #ds1.compute_bitVectorDissimilarity(B)

    #cowDataMatDict = pickle.load(open('../Experiments/cowDataMatDict_KP_aligned_june_08_2022.p', 'rb'))
    #ds1.compute_dataset_disMat(cowDataMatDict=cowDataMatDict)

    ds1.computeKPRuleStats_onDataset(optimizeForKpRules=True) #set optimizeForKpRules to False if you want to measure the true actual values from the dataset
    #set it to True if you want better limits for rule checks.
