'''
Author: Manu Ramesh

Has helper functions for tracking cows in top view.

'''

import pickle, numpy as np
import logging
from shapely.geometry import Polygon #for finding bbox centroids
from scipy import stats #for computing the bit-wise statistical Mode of bit-vectors
import pdb, pandas as pd
import os, cv2

#for matching trackPoints to tracks of multiple cows
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching, maximum_bipartite_matching

def getNewTrack(trackId=-1, forInference=False):
    #I plan to use sampleCropImg key only to make the code simpler. This key will be deleted in the final dictionary to save space.
    track = {'trackId': trackId, 'startFrame':-1, 'endFrame':-1, 'hasCow':False, 'sampleCropImg':None, 'trackPoints':[]} #trackPoints is a list of dictionaries with keys: 'videoName', 'frameNumber', 'bitVecStrList', 'rotatedBBOX_locs'

    if forInference:
        track['trackLvlPred'] = '????'
        track['predCountsDict'] = {} # To keep a running count of instance level predictions. The most frequent of these will be the track level prediction.

    return track

def getNewTrackPoint(videoName = None, frameNumber=-1, rotatedBBOX_locs=[], forInference=False): #, croppedImg=[]):
    trackPoint = {'videoName': videoName, 'frameNumber':frameNumber, 'rotatedBBOX_locs':rotatedBBOX_locs, 'bitVecStr':'', 'kpCorrectionMethodUsed':'',} #, 'croppedImg':croppedImg}
    if forInference:
        trackPoint['instLvlPred'] = '????' # to store the instance level prediction for this track point
    return trackPoint
        
def matchTrackPoints(openTracks, currentTrackPoints, closedTracks, trackId, frameCount, iouThresh=0.3, forInference=False):
    '''
    
    #Matches the unmatchedTrackPoints with the openTracks.
    #Creates new tracks for each currentTrackPt if there is no match found in the list of open tracks.
    #Closes an open track if there is no matching point for an open track.
    
    Open and closed tracks are obtained from outside the function.

    #WE USE THE MAXIMUM WEIGHT FULL BIPARTITE-MATCHING ALGORITHM HERE
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html
    
    #ROWS (m): one partition of the graph will have all the previous track points - from openTracks
    #COLUMNS (n): the other partition will have all the current track points
    #the edges will be the IOU between the rotatedBBOXes of any two track points.
    
    :param iouThresh: the minimum IOU required for a match to be considered. (In cut videos, there are abrubpt changes that result in slight overlap of prev cow with new cow. Need to avoid this.)
    :param forInference: if True, the tracks are being created for inference and not for training. Some extra keys are added to the track and trackPoint dictionaries in this case. This value is just passed to the getNewTrack function. This function does not use this value directly.

    :return: updated openTracks, closedTracks, trackId
    '''

    m = len(openTracks) #rows 1
    n = len(currentTrackPoints) #columns 2

    #create the sparse Biadjacency adjacency matrix
    #0 in the matrix will indicate that the the match is not allowed
    #a non zero (can be negative) value indicates that the match is allowed, and can be considered for final matching
    #graph = csr_array([[0]*n]*m)
    #_graph = [[0]*n]*m #changing the sparsity structure of matrix is expensive. So, we will use a list now and then make it a csr_array.
    _graph = np.zeros((m,n)) #changing one element of a list was changing multiple elements!. So I'm using an np array.

    #if frameCount == 95604:
    #            pdb.set_trace(header="Before copmputing _graph")


    for i, openTrack in enumerate(openTracks): 
        for j, currentTrackPt in enumerate(currentTrackPoints):

            #calculate IOU between the rotatedBBOXes of the two track points
            polygon1 = Polygon(openTrack['trackPoints'][-1]['rotatedBBOX_locs'])
            polygon2 = Polygon(currentTrackPt['rotatedBBOX_locs'])
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersect / (union+1e-10) #to avoid division by zero

            #_graph[i][j] = iou
            _graph[i,j] = iou 

            #if frameCount == 95604:
            #    pdb.set_trace()
                
    
    
    logging.debug(f"Matching graph: _graph = {_graph}.\nIOU Threshold = {iouThresh}\ncurrentTrackPoints = {currentTrackPoints}")
    _graph = (_graph - iouThresh).clip(0,1) #to push anything below this threshold to 0 to force a no-mach.
    
    #the min_weight_full_bipartite_matching function throws a value error even if a single row (open track) has no possible matches (all columns are 0).
    #To overcome this, we remove all such rows from _graph, store the row indices of all the other rows and replace the row indices after computing the matches.
    #For the fn to work without value error, all rows should have a match. All columns need not have a match.

    rowsWithPossibleMatches = np.where(_graph.sum(axis=1) > 0)[0] #indices of rows where IOU with at least one currentTrackPt is greater than 0
    _graph = _graph[rowsWithPossibleMatches] #this is the reduced graph with only rows that have possible matches


    graph = csr_array(_graph)

    #now match the tracks
    try:
        rowInd, colInd = min_weight_full_bipartite_matching(graph, maximize=True)

        #replacing the row indices
        rowInd = rowsWithPossibleMatches[rowInd]

    except ValueError: 
        #this occurs when all the values in the graph are 0. i.e. no match is possible.
        rowInd = []
        colInd = []

        logging.debug(f"Value error encountered (No match possible.). This is being handled by closing all open tracks.\n")
        #this will ensure all open tracks are closed, and new tracks are created for every current track point.

    #add matching points to open tracks and close tracks that have no matching points
    for r, track in enumerate(openTracks.copy()):

        if r in rowInd:
            #add matching track point
            c = colInd[rowInd.tolist().index(r)]
            track['trackPoints'].append(currentTrackPoints[c])
            if currentTrackPoints[c]['bitVecStr'] != '': track['hasCow'] = True #to mark the track as one that has a complete set of keypoints
            logging.debug(f"Adding pt to track {track['trackId']}\nNumOpenTracks = {m}, NumNewPts = {n}, rowInd = {rowInd}, colInd = {colInd}\n")

        else:
            #close the track as there are no matching points
            #If you want to, you can add a counter here to add a margin of error before closing the track.
            #you can count the number of consecutive frames without a detection and then close the track.
            #I am not doing that now as the cows could move very fast and a new cow can occupy a missed cow's spot in a matter of a few frames.
            #If that happens, the track erroneously latches on to the new cow and continues.
            openTracks.remove(track)
            track['endFrame'] = track['trackPoints'][-1]['frameNumber']
            closedTracks.append(track)                    
            logging.debug(f"Ending track {track['trackId']} as no trackPoints found in the frame {frameCount}.\nNumOpenTracks = {m}, NumNewPts = {n}, rowInd = {rowInd}, colInd = {colInd}\n")


    #create new tracks for the unmatched current track points
    for c, trackPt in enumerate(currentTrackPoints):
        if c not in colInd:

            newTrack = getNewTrack(trackId, forInference=forInference)
            newTrack['trackPoints'].append(trackPt)
            newTrack['startFrame'] = trackPt['frameNumber']
            if trackPt['bitVecStr'] != '': newTrack['hasCow'] = True #to mark the track as one that has a complete set of keypoints
            openTracks.append(newTrack)

            logging.debug(f"Starting new track {newTrack['trackId']}\nNumOpenTracks = {m}, NumNewPts = {n}, rowInd = {rowInd}, colInd = {colInd}\n")

            trackId += 1


    return openTracks, closedTracks, trackId

