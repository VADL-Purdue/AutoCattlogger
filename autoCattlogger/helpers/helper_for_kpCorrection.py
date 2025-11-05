'''
Author: Manu Ramesh

Code should take in YAML stat file and generate affected KPs list for each rule.

REFACTORING WARNING:
This module was originally called experiment_getAffectedKPs(.py).
This file is being promoted from experiments folder to Eidetic Cattle Recognition folder.
The module will henceforth be called helper_for_kpCorrection.
There used to be a module with the name helper_for_kpCorrection before, but that is now called helper_for_kpInterpolation.
All keypoint correction codes are here, not in the module meant for interpolation.

The keypoint correction strategy of first invoking iterativeMode and then SBF mode is coded in autoCattlogger(.py) module.

'''
import yaml, os
try:
    from ndicts.ndicts import DataDict, NestedDict
except ImportError:
    from ndicts import DataDict, NestedDict

import sys, pdb
import numpy as np

sys.path.insert(0, '../../') #top level dir
from autoCattlogger.helpers.helper_for_infer import cow_tv_KP_rule_checker, cow_tv_KP_rule_checker2
from autoCattlogger.helpers.helper_for_kpInterpolation import interpolateKPs

# Function which returns subset or r length from n
from itertools import combinations # https://www.geeksforgeeks.org/itertools-combinations-module-python-print-possible-combinations/


def generateAffectedKPsYAML(statFilePath, saveDir):
    '''
    Generates YAML file of lists of affected KPs for each rule (if broken). Saves the list a s YAML File.
    :param statFilePath: path to YAML stat file
    :param saveDir: path to dir in which output is saved
    :return:
    '''

    statsDict = yaml.safe_load(open(statFilePath, 'r'))
    statsNDict = NestedDict(statsDict)
    affectedKPsNDict = statsNDict.copy()

    #look up table (LUT) for keypoint names
    #kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"] # for reference
    kpn_lookup = {'lshldr':["left_shoulder"], 'withers': ["withers"],'rshldr':["right_shoulder"],'cback':["center_back"],'lhip':["left_hip_bone"], 'hcon':["hip_connector"],
                    'rhip':["right_hip_bone"], 'lpin':["left_pin_bone"], 'thead': ["tail_head"], 'rpin':["right_pin_bone"],
                  'pins': ["left_pin_bone","right_pin_bone"], 'hips':["left_hip_bone","right_hip_bone"], 'shldrs':["left_shoulder","right_shoulder"],
                  'tailHead':["tail_head"], 'hipCon':["hip_connector"],
                  "left_shoulder":["left_shoulder"], "withers":["withers"], "right_shoulder":["right_shoulder"], "center_back":["center_back"], "left_hip_bone":["left_hip_bone"],
                  "hip_connector":["hip_connector"], "right_hip_bone":["right_hip_bone"], "left_pin_bone":["left_pin_bone"], "tail_head":["tail_head"]
                  }

    for key, val in statsNDict.items():
        lowestLevelKey = key[-1]
        affectedKPs = []

        for shortName, fullNamesList in kpn_lookup.items():
            if shortName in lowestLevelKey:
                for name in fullNamesList:
                    if name not in affectedKPs:
                        affectedKPs.append(name)

        #print(f"Affected KPs = {affectedKPs}")
        affectedKPsNDict[key] = affectedKPs

    affectedKPsDict = affectedKPsNDict.to_dict()
    print(f"Affected KPs NDict = {affectedKPsDict}")

    outFilePath = saveDir + os.path.split(statFilePath)[-1].split('.')[0] + "_affectedKPs.yml"
    print(f"\nAffected KPs yml created at: {outFilePath}")
    yaml.dump(affectedKPsDict, open(outFilePath, 'w'))


def computeWeightsForMisplacedKPs(affectedKPsDict, keypoint_names):
    '''
    Computes weights for misplaced Keypoints. These weights are reciprocals of number of rules in which each keypoint appears.
    :param affectedKPsDict: pass the dict after extracting from the yaml file
        dictionary of the form - rule:[affectedKP1, affectedKP2, affectedKP3 ....]
    :param keypoint_names: list of keypoint names
    :return:
    '''
    affectedKPsNDict = NestedDict(affectedKPsDict)
    kpn = keypoint_names

    countsDict = dict(zip(kpn, [0]*len(kpn)))

    for rule, affectedKPs in affectedKPsNDict.items():
            for affectedKP in affectedKPs:
                countsDict[affectedKP] += 1

    print(f"Rule counts for each keypoint = {countsDict}")

    weightsDict = countsDict.copy()
    for k, v in countsDict.items():
        weightsDict[k] = 1/v #you could try other functions later if needed

    print(f"Weights for each keypoint = {weightsDict}")

    return countsDict, weightsDict

def predictMisplacedKeypoints(instanceRulePassesDict, affectedKPsDict, kp_weights_dict=None, return_hitsDicts=False, printDebugInfoToScreen=False):
    '''
    Predicts which keypoints are misplaced using a dictionary of rule passes and another that tells which keypoints are affected for each rule broken.
    :param instanceRulePassesDict:
    :param affectedKPsDict:
    :param kp_weights_dict: Dictionary of weights for each keypoint.
    :param return_hitsDicts: if True, returns hitsDicts
    :return:
    '''

    kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]
    hitsDict = {"left_shoulder":0, "withers":0, "right_shoulder":0, "center_back":0, "left_hip_bone":0, "hip_connector":0, "right_hip_bone":0, "left_pin_bone":0, "tail_head":0, "right_pin_bone":0} #counts number of times a kp is affected
    weighted_hitsDict = hitsDict.copy()

    sideKpMirrors = {#"left_shoulder":"right_shoulder", "right_shoulder":"left_shoulder", #there are no interpolation policies currently to support independent interpolation of shoulders - so, commenting this pair
                     "left_hip_bone":"right_hip_bone", "right_hip_bone":"left_hip_bone",
                     "left_pin_bone":"right_pin_bone", "right_pin_bone":"left_pin_bone"} #the opposite side points
    # sideKpMirrors and partnersInCrimeKPs specify which other keypoint should also be removed. worstKP:secondWorstKP
    # eg - if hipCon is bad, we need to fix it first and then the cback. The interpolation function takes care of this order.
    # but both should be made invisible
    partnersInCrimeKPs = {"center_back":"hip_connector", "right_hip_bone":"hip_connector", "left_hip_bone":"hip_connector",
                          "tail_head":"hip_connector"}

    instanceRulePassesNDict = NestedDict(instanceRulePassesDict)
    affectedKPsNDict = NestedDict(affectedKPsDict)

    for rule, state in instanceRulePassesNDict.items():
        if state == 'Fail':
            for kp in affectedKPsNDict[rule]:
                hitsDict[kp] += 1
                weighted_hitsDict[kp] += 1 * kp_weights_dict[kp] if kp_weights_dict is not None else 1 #all weights are assumed to be 1 if kp_weights is None

    hitsDictSafe = hitsDict.copy()
    weighted_hitsDictSafe = weighted_hitsDict.copy() # we will alter weighted_hitsDict later, so we save a backup now
    if printDebugInfoToScreen: print(f"Weighted Hits Dict = {weighted_hitsDictSafe}")


    predMisplacedKpList = [max(weighted_hitsDict, key=lambda x: weighted_hitsDict[x])] #the first misplaced kp

    if hitsDict[predMisplacedKpList[0]] == 0: # if all KPs have 0 hits
        predMisplacedKpList = None

    else:
        #check if there are other keypoints that are simultaneously misplaced



        weighted_hitsDict.pop(predMisplacedKpList[0])  # remove top element, returns value (not key) - this operation alters the original dictionary - that is why we took a backup
        worstKP = predMisplacedKpList[0]
        secondWorstKP = max(weighted_hitsDict, key=lambda x: weighted_hitsDict[x])  # max with original max removed

        # some hard coding
        if instanceRulePassesDict["minStatDict"]["fractional_lengths"]["flen_lpin_tailHead"] == "Fail" or instanceRulePassesDict["minStatDict"]["fractional_lengths"]["flen_rpin_tailHead"] == "Fail": #otherwise, tail head is not being included.
            if "tail_head" not in predMisplacedKpList:
                predMisplacedKpList.insert(0, "tail_head") #add tail head to first position - (actually position does not matter)
                # with this you can see that if tail head does not have max weighted hits and is yet in the top position, it is because of this hard coding

        # works only for left and right corresponding pairs for now
        elif worstKP in sideKpMirrors and secondWorstKP == sideKpMirrors[worstKP]: # worst kp is a side kp and corresponding keypoint on the other side is also misplaced simultaneously
            predMisplacedKpList.append(secondWorstKP)
        elif worstKP in partnersInCrimeKPs and secondWorstKP == partnersInCrimeKPs[worstKP]: #other kp should also be removed to enforce interpolation order
            predMisplacedKpList.append(secondWorstKP)

    if printDebugInfoToScreen: print(f"Predicted Misplaced Keypoints: {predMisplacedKpList}")



    if return_hitsDicts:
        return predMisplacedKpList, hitsDictSafe, weighted_hitsDictSafe

    else:
        return predMisplacedKpList


def correctMispalcedKPs_iterativeMode(kpts, maskContour, datasetKpStatsDict, affectedKPsDict, kpWeightsDict, keypoint_names, keypoint_threshold=0.5, cowW=350, cowH=750, frameW=1920, frameH=1080, printDebugInfoToScreen=False):
    '''
    iteratively corrects the top predicted misplaced keypoint
    :param kpts: Keypoints array [[x1,y1,visibility1] [x2,y2,vis2],...]
    '''

    initial_kpts = kpts.copy()

    kpn = keypoint_names

    previousTotalHits = 1e5
    totalHits = 999
    cow_tv_KP_confidence1 = cow_tv_KP_confidence2 = None
    cow_tv_KP_maxConfidence1 = cow_tv_KP_maxConfidence2 = None

    while totalHits < previousTotalHits:

        previousTotalHits = totalHits

        cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1 = cow_tv_KP_rule_checker(keypoints=kpts, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH)
        if printDebugInfoToScreen: print(f"\nRuleChecker1: Current KP_Confidence = {cow_tv_KP_confidence1}/{cow_tv_KP_maxConfidence1}\n")

        cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2, instanceRulePassesDict, instanceKpStatsDict = cow_tv_KP_rule_checker2(keypoints=kpts, maskContour=maskContour, datasetKpStatsDict=datasetKpStatsDict, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH, returnInstanceKpStatsDict=True)
        if printDebugInfoToScreen: print(f"RuleChecker2: Current KP_Confidence = {cow_tv_KP_confidence2}/{cow_tv_KP_maxConfidence2}\n")

        predMisplacedKpList, hitsDict, weightedHitsDict = predictMisplacedKeypoints(instanceRulePassesDict=instanceRulePassesDict, affectedKPsDict=affectedKPsDict, kp_weights_dict=kpWeightsDict, return_hitsDicts=True)
        if printDebugInfoToScreen: print(f"Current predicted misplaced KP(s) = {predMisplacedKpList}")
        totalHits = sum([x[1] for x in hitsDict.items()])

        if predMisplacedKpList != None:  # i.e. total hits != 0
            for predMisplacedKp in predMisplacedKpList:  # all misplaced keypoints should be made invisible at the same time!
                if printDebugInfoToScreen: print(f"Correcting keypoint: {predMisplacedKp}")
                kpts[keypoint_names.index(predMisplacedKp)] = keypoint_threshold - 0.2

            kpts = interpolateKPs(instance_KPs=kpts, keypoint_names=keypoint_names, keypoint_threshold=0.5, frameW=frameW, frameH=frameH)  # using this to interpolate missing kps

            # save instace keypoint rule measurements in a yaml file
            # yaml.dump(instanceKpStatsDict, open('./instanceKpStatsDict.yml', 'w'))

            # save instance rule passes in a yaml file - for rule checker2
            # yaml.dump(instanceRulePassesDict, open('./instanceRulePassesDict.yml', 'w'))

            predMisplacedKpList, hitsDict, weightedHitsDict = predictMisplacedKeypoints(instanceRulePassesDict=instanceRulePassesDict, affectedKPsDict=affectedKPsDict, kp_weights_dict=kpWeightsDict, return_hitsDicts=True)

            if printDebugInfoToScreen: print(f"\nTotal hits = {totalHits}, previous total hits = {previousTotalHits}")

        else:
            if printDebugInfoToScreen: print(f"predMisplacedKpList is None! Breaking out.")
            break

    allRulesPassed = False #a variable that tells if all keypoints are conforming
    if cow_tv_KP_confidence1 == cow_tv_KP_maxConfidence1 and cow_tv_KP_confidence2 == cow_tv_KP_maxConfidence2 and cow_tv_KP_maxConfidence1 != 0:
        allRulesPassed = True
        if printDebugInfoToScreen: print(f"All rules passed. Keypoint errors are fixed.")

    correctedKPs = [] #list of names of corrected keypoints
    for idx, kpRow in enumerate(initial_kpts):
        if not np.array_equal(kpts[idx], kpRow):
            correctedKPs.append(keypoint_names[idx])

    return allRulesPassed, kpts, correctedKPs, (cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1), (cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2)


def correctMispalcedKPs_StrategicBruteForceMode(kpts, maskContour, datasetKpStatsDict, affectedKPsDict, kpWeightsDict, keypoint_names, keypoint_threshold=0.5,  cowW=350, cowH=750, frameW=1920, frameH=1080, printDebugInfoToScreen=False):
    '''
    Strategic Brute Force - attempts to fix all kps with non zero hits
    -   one KP at a time
    -   top 3 worst hit KPs two at a time
    -   top 3 worst hit KPs all at once

    :param kpts: Keypoints array [[x1,y1,visibility1] [x2,y2,vis2],...]
    :param keypoint_names:
    :param keypoint_threshold:
    :return:
    '''

    initial_kpts = kpts.copy()

    kpn = keypoint_names

    cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1 = cow_tv_KP_rule_checker(keypoints=kpts, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH)
    if printDebugInfoToScreen: print(f"\nRuleChecker1: Current KP_Confidence = {cow_tv_KP_confidence1}/{cow_tv_KP_maxConfidence1}\n")

    cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2, instanceRulePassesDict, instanceKpStatsDict = cow_tv_KP_rule_checker2(keypoints=kpts, maskContour=maskContour, datasetKpStatsDict=datasetKpStatsDict, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH, returnInstanceKpStatsDict=True)
    if printDebugInfoToScreen: print(f"RuleChecker2: Current KP_Confidence = {cow_tv_KP_confidence2}/{cow_tv_KP_maxConfidence2}\n")

    predMisplacedKpList, hitsDict, weightedHitsDict = predictMisplacedKeypoints(instanceRulePassesDict=instanceRulePassesDict, affectedKPsDict=affectedKPsDict, kp_weights_dict=kpWeightsDict,return_hitsDicts=True)

    # for all kps with non zero hits, try to use interpolated kps in their place and see if they pass all rules
    kpts_temp = kpts.copy()
    fixed = False # all errors fixed/corrected

    sorted_kps_byWeigtedHits = sorted(weightedHitsDict.items(), key=lambda x: x[1], reverse=True)  # list of tuples sorted in descending order of weighted hits
    if printDebugInfoToScreen: print(f"Sorted kps by weighted hits = {sorted_kps_byWeigtedHits}")

    # for kp, hits in weightedHitsDict.items():
    for kp, hits in sorted_kps_byWeigtedHits:
        if hits != 0:
            if printDebugInfoToScreen: print(f"*********************************************************************************************")
            if printDebugInfoToScreen: print(f"Strategic Brute Force: Correcting keypoint: {kp}")
            kpts_temp[keypoint_names.index(kp)] = keypoint_threshold - 0.2
            kpts_temp = interpolateKPs(instance_KPs=kpts_temp, keypoint_names=keypoint_names, keypoint_threshold=0.5,
                                       frameW=frameW, frameH=frameH)  # using this to interpolate missing kps

            cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1 = cow_tv_KP_rule_checker(keypoints=kpts_temp, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH)
            if printDebugInfoToScreen: print(f"\nRuleChecker1: Current KP_Confidence = {cow_tv_KP_confidence1}/{cow_tv_KP_maxConfidence1}\n")

            cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2, instanceRulePassesDict, instanceKpStatsDict = cow_tv_KP_rule_checker2(keypoints=kpts_temp, maskContour=maskContour, datasetKpStatsDict=datasetKpStatsDict, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH, returnInstanceKpStatsDict=True)
            if printDebugInfoToScreen: print(f"RuleChecker2: Current KP_Confidence = {cow_tv_KP_confidence2}/{cow_tv_KP_maxConfidence2}\n")

            if (cow_tv_KP_confidence2 == cow_tv_KP_maxConfidence2) and cow_tv_KP_maxConfidence2 != 0 and (cow_tv_KP_confidence1 == cow_tv_KP_maxConfidence1):  # and cow_tv_KP_maxConfidence1 != 0: -> both max confidences are 0 at the same time, checking for one is enough
                # best possible prediction of misplaced KP
                fixed = True
                kpts = kpts_temp
                if printDebugInfoToScreen: print(f"All rules passed! Breaking out of Strategic Brute Force Correction. Corrected KP = {kp}")

                # save instace keypoint rule measurements in the yaml file
                # yaml.dump(instanceKpStatsDict, open('./instanceKpStatsDict.yml', 'w'))

                # pdb.set_trace()
                break

            # else: # else by default
            kpts_temp = kpts.copy()

    if fixed == False:  # if it is still not fixed...,
        # try to eliminate worst two kps

        possibleCombos = list(combinations(sorted_kps_byWeigtedHits[:3], r=2))  # returns all nCr possible combos - 3C2 of the top 3 worst Kps
        possibleCombos.append(
            sorted_kps_byWeigtedHits[:3])  # adding all three worst KPs to be eliminated at the same time

        for possibleCombo in possibleCombos:
            if printDebugInfoToScreen: print(f"*********************************************************************************************")
            if printDebugInfoToScreen: print(f"Strategic Brute Force: Correcting keypoint combos: {[x[0] for x in possibleCombo]}")
            for kp, hits in possibleCombo:  # eliminate each KP in the combo
                # if printDebugInfoToScreen: print(f"Eliminating {kp}")
                kpts_temp[keypoint_names.index(kp)] = keypoint_threshold - 0.2

            kpts_temp = interpolateKPs(instance_KPs=kpts_temp, keypoint_names=keypoint_names, keypoint_threshold=0.5, frameW=frameW, frameH=frameH)  # using this to interpolate missing kps

            cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1 = cow_tv_KP_rule_checker(keypoints=kpts_temp, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH)
            if printDebugInfoToScreen: print(f"\nRuleChecker1: Current KP_Confidence = {cow_tv_KP_confidence1}/{cow_tv_KP_maxConfidence1}\n")

            cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2, instanceRulePassesDict, instanceKpStatsDict = cow_tv_KP_rule_checker2(keypoints=kpts_temp, maskContour=maskContour, datasetKpStatsDict=datasetKpStatsDict, keypoint_names=kpn, keypoint_threshold=0.5, cowW=cowW, cowH=cowH, frameW=frameW, frameH=frameH, returnInstanceKpStatsDict=True)
            if printDebugInfoToScreen: print(f"RuleChecker2: Current KP_Confidence = {cow_tv_KP_confidence2}/{cow_tv_KP_maxConfidence2}\n")

            if (cow_tv_KP_confidence2 == cow_tv_KP_maxConfidence2) and cow_tv_KP_maxConfidence2 != 0 and (cow_tv_KP_confidence1 == cow_tv_KP_maxConfidence1):  # and cow_tv_KP_maxConfidence1 != 0: -> both max confidences are 0 at the same time, checking for one is enough
                # best possible prediction of misplaced KP
                fixed = True
                kpts = kpts_temp
                if printDebugInfoToScreen: print(f"All rules passed! Breaking out of Strategic Brute Force Correction. Corrected KPs = {[x[0] for x in possibleCombo]}")

                # save instace keypoint rule measurements in the yaml file
                # yaml.dump(instanceKpStatsDict, open('./instanceKpStatsDict.yml', 'w'))

                # pdb.set_trace()
                break

                # else: # else by default
            kpts_temp = kpts.copy()

    allRulesPassed = fixed

    if allRulesPassed == False:  # if it is still not fixed
        kpts_temp = kpts.copy()
        if printDebugInfoToScreen: print(f"Strategic Brute Force failed to fix keypoints. (Too many KPs misplaced).")

    correctedKPs = [] #list of names of corrected keypoints
    for idx, kpRow in enumerate(initial_kpts):
        if not np.array_equal(kpts[idx], kpRow):
            correctedKPs.append(keypoint_names[idx])

    return allRulesPassed, kpts, correctedKPs, (cow_tv_KP_confidence1, cow_tv_KP_maxConfidence1), (cow_tv_KP_confidence2, cow_tv_KP_maxConfidence2)


if __name__ == "__main__":

    statFilePath = '../Eidetic Cattle Recognition/output/statOutputs/datasetKpStatsDict_kp_dataset_v4_train.yml'
    saveDir = "../Eidetic Cattle Recognition/output/statOutputs/"
    generateAffectedKPsYAML(statFilePath=statFilePath, saveDir=saveDir)

    #affectedKPsDict = yaml.safe_load(open('../Eidetic Cattle Recognition/output/statOutputs/datasetKpStatsDict_kp_dataset_v4_train_affectedKPs.yml', 'r'))
    #kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]
    #_, _ = computeWeightsForMisplacedKPs(affectedKPsDict=affectedKPsDict, keypoint_names=kpn)


