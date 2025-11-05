'''
Author: Manu Ramesh

Tool to generate the stats - (upper and lower bounds) for the rules checker from the given dataset.
The dataset must be in COCO format.
'''

import os, sys, pickle, argparse, pdb
sys.path.insert(0, '../../') #Top level dir of the repo
from autoCattlogger.helpers.helper_for_stats import DatasetStats


def generateStats(args):
    ds1 = DatasetStats(dsetFilePath=args.annotationFilePath)
    ds1.computeKPRuleStats_onDataset(optimizeForKpRules= not args.doNotOptimizeForKpRules, outRootDir=args.outRootDir) #set optimizeForKpRules to False if you want to measure the true actual values from the dataset
    #set it to True if you want better limits for rule checks.

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--annotationFilePath', type=str, required=True, help='path to the COCO format annotation file')
    parser.add_argument('-n', '--doNotOptimizeForKpRules', action='store_true', help='set to True if you do not want to optimize the stats for keypoint rules')
    parser.add_argument('-o', '--outRootDir', type=str, default='./', help='root directory to save the stats files')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()
    generateStats(args)