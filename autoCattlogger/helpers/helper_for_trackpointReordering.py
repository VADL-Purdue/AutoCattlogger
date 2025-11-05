'''
Author: Manu Ramesh

Has helper functions for reordering track points in order of frames instead of tracks.
This is helpful in cases where the videos use encoding standards such as H.264 that do not support accurate seeking.
Then, we can process all cows in a frame at a time, going frame by frame, rather than seeking to each trackpoint's frame, going track by track.
'''

from tqdm import tqdm
import pdb


def createFrameWiseTracksInfo(tracks, printDebugInfoToScreen=False):

    #Track keys: dict_keys(['trackId', 'startFrame', 'endFrame', 'hasCow', 'trackPoints', 'entryDirection', 'exitDirection', 'autoCattBitVecStr', 'gt_label'])
    #Track point keys: dict_keys(['videoName', 'frameNumber', 'rotatedBBOX_locs', 'bitVecStr', 'kpCorrectionMethodUsed'])

    # Get a list of all video names in the tracks
    frameWiseTracksInfo = {}

    videoNames = []
    for track in tracks:
        for trackPoint in track['trackPoints']:
            videoName = trackPoint['videoName']
            if videoName not in videoNames:
                videoNames.append(videoName)
    
    for videoName in videoNames:

        if printDebugInfoToScreen: print(f"Processing video: {videoName}")

        frameWiseInfo = {}

        for track in tqdm(tracks, desc=f"Tracks for {videoName}"):
            gt_label = track['gt_label']
            for trackPoint in track['trackPoints']:
                if trackPoint['videoName'] == videoName:
                    frameNumber = trackPoint['frameNumber']
                    rotatedBBOX_locs = trackPoint['rotatedBBOX_locs']
                    bitVecStr = trackPoint['bitVecStr']

                    info = {'gt_label': gt_label, 'rotatedBBOX_locs': rotatedBBOX_locs, 'bitVecStr': bitVecStr}
                    if frameNumber not in frameWiseInfo:
                        frameWiseInfo[frameNumber] = [info]
                    else:
                        frameWiseInfo[frameNumber].append(info)

            # if printDebugInfoToScreen: print(f"Processed track {track['trackId']} for video {videoName}.")
            # pdb.set_trace()
        
        # For ease of access, create a list of frames and save it in the dictionary
        frameList = sorted(frameWiseInfo.keys())
        frameWiseTracksInfo[videoName] = {'orderedFramesList': frameList, 'frameWiseInfo': frameWiseInfo}
        if printDebugInfoToScreen: print(f"Processed video {videoName} with {len(frameList)} frames.")

    # pdb.set_trace(header='Processed all tracks. Check the frameWiseTracksInfo dictionary.')
    #Format: frameWiseTracksInfo['cam24_2022-06-08_05-29-20.avi']['frameWiseInfo'][88402]


    return frameWiseTracksInfo