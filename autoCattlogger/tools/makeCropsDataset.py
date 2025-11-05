'''
Author: Manu Ramesh

Saves the cropped images of cows from the tracks to create a cropped image dataset.
We use this to train the cow identifiers from other works in literature.
'''

import pickle, pdb, glob, os, sys, cv2, pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import numpy as np

sys.path.insert(0, '../../') # top level directory

from autoCattleID.helpers.helper_for_trackPointFiltering import getFilteredCattlog, getFilterFnFromName
from autoCattlogger.helpers.helper_for_trackpointReordering import createFrameWiseTracksInfo

def get_cowCroppedImg(frame_number=None, video_path=None, rotatedBBOX_locs=None, frame=None):
    """
    Gets the cow oriented cropped image from video frame.

    Args:
        frame_number (int): The frame number to extract.
        video_path (str): Path to the video file.
        rotatedBBOX_locs: the rotated bbox corner points. This is a 4x2 array of points. If None, will get the points from the trackPoint.
        frame (optional): If provided, uses this frame instead of reading from the video file.

    Returns:
        
    """
    
    if frame is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)  # Set the frame position (0-indexed)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Unable to retrieve frame {frame_number} from {video_path}")
            return None

    '''
    Rotated bbox locs are in the format: ABCD [[xA, yA], [xB, yB], [xC, yC], [xD, yD]]
     A--B
     |  |
     D--C
     '''

    bboxW = np.linalg.norm(rotatedBBOX_locs[0] - rotatedBBOX_locs[1] + 1) #+1 to include the last pixel
    bboxH = np.linalg.norm(rotatedBBOX_locs[0] - rotatedBBOX_locs[3] + 1) #+1 to include the last pixel
    #bboxAspectRatio = bboxW / bboxH

    #warp bbox to cropped image
    destination_points = np.array([[0, 0], [bboxW, 0], [bboxW, bboxH], [0, bboxH]], dtype="float32")
    M = cv2.getPerspectiveTransform(rotatedBBOX_locs.astype(np.float32), destination_points)

    cropped_img = cv2.warpPerspective(frame, M, (int(bboxW), int(bboxH)), flags=cv2.INTER_LINEAR)

    #save only horizontal cow images for uniformity
    h,w = cropped_img.shape[:2]
    if w<h:
        cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)

    return cropped_img

def save_cropsFromTracks(track, srcVidDir, outRootDir, filterTrackPts=False, filterFnName='trackPtsFilter_byProximityOfRBboxToFrameCenter'):
    '''
    Fetches the crops from the track and saves them to the outRootDir.

    Inputs:
        track (dict): A dictionary containing the track information.
        srcVidDir (str): The directory containing the video files.
        outRootDir (str): The directory to save the cropped images.
        filterTrackPts (bool): Whether to filter the track points.
        filterFnName (str): The name of the filter function to use if filterTrackPts is True.

    Outputs:
        outVal (dict): A dictionary containing the cow ID and the number of images saved.
    '''

    os.makedirs(f"{outRootDir}/{track['gt_label']}/", exist_ok=True)



    trackPoints = [x for x in track['trackPoints'] if x['bitVecStr'] != ''] #Get a full cow
    
    
    if filterTrackPts:
        trackPtsFilterFn = getFilterFnFromName(filterFnName) # will be None if filterFnName is None
        if trackPtsFilterFn is not None:
            trackPoints = trackPtsFilterFn(trackPoints, inclusionPercentage_top=0.2, frameW=1920, frameH=1080) #Filter track points
        else:
            raise ValueError(f"Filter function name {filterFnName} is not recognized. Please provide a valid filter function name from among {getFilterFnFromName(None, True)}.")

    #must return a dict with the label and the number of images saved.
    #this will be useful info, we could plot a histogram of the number of images
    outVal = {'cowID':track['gt_label'], 'nTrackPoints':len(trackPoints)}

    for trackPt in trackPoints:
        
        frame_number = trackPt['frameNumber']
        videoName = trackPt['videoName']
        videoPath = f"{srcVidDir}/{videoName}"
        rotatedBBOX_locs = trackPt['rotatedBBOX_locs']

        # Get the cow cropped image
        cow_cropped_img = get_cowCroppedImg(frame_number, videoPath, rotatedBBOX_locs)

        if cow_cropped_img is not None:
            # Save the cropped image

            outImgName = f"{track['gt_label']}_{videoName.split('.')[0]}_{frame_number}.jpg"
            outImgPath = f"{outRootDir}/{track['gt_label']}/{outImgName}"
            cv2.imwrite(outImgPath, cow_cropped_img)

    return outVal

def saveCropsFromVideos_framewise(tracks, videoName, srcVidDir, outRootDir, requiredCowIDs, filterTrackPts=False, filterFnName='trackPtsFilter_byProximityOfRBboxToFrameCenter'):
    '''
    Iterates over every frame of the video and saves the crops from the track points in the frame.
    To avoid seeking errors in H264 videos.

    (frameWiseTracksInfo: frameWiseTracksInfo dict for the spcific videoName. Dict {'orderedFramesList': [frameNumbers], 'frameWiseInfo': {frameNumber: [{trackPointInfo}, ...]}})

    Inputs:
        tracks (list): List of tracks.
        videoName (str): Name of the video.
        srcVidDir (str): The directory containing the video files.
        outRootDir (str): The directory to save the cropped images.
        requiredCowIDs (list): List of cow IDs to include in the dataset.
        filterTrackPts (bool): Whether to filter the track points.
        filterFnName (str): The name of the filter function to use if filterTrackPts is True.

    Outputs:
        imageCounts (dict): A dictionary containing the cow ID and the number of images saved.
    '''


    # allTrackPoints = [x for y in frameWiseTracksInfo.values() for x in y]
    
    # Filter trackPoints if necessary to create a new frame wise info dict
    if filterTrackPts:
        trackPtsFilterFn = getFilterFnFromName(filterFnName) # will be None if filterFnName is None
        if trackPtsFilterFn is not None:
            for track in tracks:
                trackPoints = track['trackPoints']
                trackPoints = trackPtsFilterFn(trackPoints, inclusionPercentage_top=0.2, frameW=1920, frameH=1080) #Filter track points
                track['trackPoints'] = trackPoints #filtered track points
        else:
            raise ValueError(f"Filter function name {filterFnName} is not recognized. Please provide a valid filter function name from among {getFilterFnFromName(None, True)}.")


    frameWiseTrackInfo_allVideos = createFrameWiseTracksInfo(tracks, printDebugInfoToScreen=False)

    if videoName in frameWiseTrackInfo_allVideos:
        frameWiseTracksInfo = frameWiseTrackInfo_allVideos[videoName] #Creating frameWiseTracksInfo for the specific videoName, using the filtered track points
    else:
        #All trackpoints in this video are filtered out
        print(f"All track points in video {videoName} are filtered out. Skipping this video.")
        frameWiseTracksInfo = {'orderedFramesList': [-1], 'frameWiseInfo': {}} #Set condition for the while loop below to break

    
        

    videoPath = f"{srcVidDir}/{videoName}"
    cap = cv2.VideoCapture(videoPath)
    
    frameCount = 0 #This is 1 indexed
    nextFrameWithCow_idx = 0
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video

    #must return a dict with the label and the number of images saved.
    #this will be useful info, we could plot a histogram of the number of images
    # outVal = {'cowID':track['gt_label'], 'nTrackPoints':len(trackPoints)}
    imageCounts = {}

    pbar = tqdm(total=totalFrames, desc=f"Processing frames for {videoName}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1
        pbar.update(1)

        #Skip the frame till you reach the next frame with cow
        orderedFramesList = frameWiseTracksInfo['orderedFramesList']
        nextFrameWithCow = orderedFramesList[nextFrameWithCow_idx]
        pbar.set_description(f"Processing frames (next cow at frame {nextFrameWithCow})")

        if frameCount < nextFrameWithCow:
            continue

        # If we are done processing all frames with cows, break the loop
        if frameCount > nextFrameWithCow:
            pbar.update(totalFrames - frameCount)  # Update the progress bar to the end
            break

        trackPtsList = frameWiseTracksInfo['frameWiseInfo'][frameCount]

        # # Filter trackPoints if necessary
        # if filterTrackPts and trackPtsFilterFn is not None:
        #     #these trackPts might have some values missing. But they do have the rotatedBBOX_locs required by the filterFn
        #     trackPtsList = trackPtsFilterFn(trackPtsList, inclusionPercentage_top=0.2, frameW=1920, frameH=1080) 


        for trackPtInfo in trackPtsList:
            bitVecStr = trackPtInfo['bitVecStr']
            if bitVecStr == '':
                #not all keypoints are detected
                #include only full cows
                continue
            
            gt_label = trackPtInfo['gt_label']
            rotatedBBOX_locs = trackPtInfo['rotatedBBOX_locs']

            if gt_label not in requiredCowIDs:
                #skip the cow if it is not in the required cow IDs
                # print(f"Skipping cow {gt_label} in video {videoName} at frame {frameCount}. Not in required cow IDs.")
                continue

            # Get the cow cropped image
            cow_cropped_img = get_cowCroppedImg(rotatedBBOX_locs=rotatedBBOX_locs, frame=frame)

            if cow_cropped_img is not None:
                # Save the cropped image
                outImgName = f"{gt_label}_{videoName.split('.')[0]}_{frameCount}.jpg"
                outImgPath = f"{outRootDir}/{gt_label}/{outImgName}"
                os.makedirs(os.path.dirname(outImgPath), exist_ok=True)
                cv2.imwrite(outImgPath, cow_cropped_img)

                if gt_label not in imageCounts:
                    imageCounts[gt_label] = 0
                imageCounts[gt_label] += 1

        nextFrameWithCow_idx = min(nextFrameWithCow_idx + 1, len(orderedFramesList) - 1)  # Move to the next frame with cow in the ordered frames list
    
    cap.release()
    pbar.close()

    return imageCounts


def create_cropsDataset_trackWise(tracks, srcVidDir, filterTrackPts=False, filterFnName='trackPtsFilter_byProximityOfRBboxToFrameCenter'):
    '''
    Create crops dataset from tracks. Seeks to the frame number (saved in track point info) for each track point and saves the cropped image.
    This method is faster but may not be accurate with H.264 type of encoding that do not support accurate seeking.

    This method is fast because it directly seeks to the required frame and does not waste time decoding unnecessary frames.
    
    Inputs:
        tracks (list): List of tracks.
        srcVidDir (str): The directory containing the source video files.
        filterTrackPts (bool): Whether to filter the track points.
        filterFnName (str): The name of the filter function to use if filterTrackPts is True.

    Outputs:
        None
        Saves the cropped images to the outRootDir.
    '''

    outRootDir = f"./CropsDataset_trackWise_{'_filtered_'+filterFnName if filterTrackPts else '_unfiltered'}/"
    os.makedirs(outRootDir, exist_ok=True)

    # Process each track in parallel
    print(f"Processing tracks...")
    with Pool(cpu_count()-1) as pool:
        results = pool.starmap(save_cropsFromTracks, tqdm([(track, srcVidDir, outRootDir, filterTrackPts, filterFnName) for track in tracks], total=len(tracks), desc=f"Processing tracks.", unit="track"))
        # TQDM Progress bar with starmap https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    
    
    # Save the results to a file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{outRootDir}/crops_dataset_summary{'_filtered' if filterTrackPts else '_unfiltered'}.csv", index=False)

    print(f"Processed {len(tracks)} tracks. Cropped images saved to {outRootDir}.")


def create_cropsDataset_frameWise(tracks, srcVidDir, filterTrackPts=False, filterFnName='trackPtsFilter_byProximityOfRBboxToFrameCenter'):
    '''
    Create crops dataset from tracks. Iterates over every frame of the video and saves the crops from the track points in the frame.
    Use this method if you are dealing with H.264 type of encoding that do not support accurate seeking.

    This method is slower because it iterates over every frame of the video and decodes them, even if it does not have any cows in it.
    
    Inputs:
        tracks (list): List of tracks.
        srcVidDir (str): The directory containing the source video files.
        filterTrackPts (bool): Whether to filter the track points.
        filterFnName (str): The name of the filter function to use if filterTrackPts is True.

    Outputs:
        None
        Saves the cropped images to the outRootDir.
    '''

    outRootDir = f"./CropsDataset_frameWise_{'_filtered_'+filterFnName if filterTrackPts else '_unfiltered'}/"
    os.makedirs(outRootDir, exist_ok=True)


    trackInfo_frameWise = createFrameWiseTracksInfo(tracks, printDebugInfoToScreen=False) # create this once


    cow_ids = [track['gt_label'] for track in tracks]
    # cow_ids = set() #cow ids of the day being processed
    # for videoName in trackInfo_frameWise.keys():
    #     cow_ids |= set([x['gt_label'] for y in trackInfo_frameWise[videoName]['frameWiseInfo'].values() for x in y])


    results = []
    # for videoName in trackInfo_frameWise.keys():
    #     results.append(saveCropsFromVideos_framewise(tracks, videoName, srcVidDir, outRootDir, cow_ids, filterTrackPts=filterTrackPts, filterFnName=filterFnName))

    #process each video in parallel
    with Pool(cpu_count()-1) as pool:
        results = pool.starmap(saveCropsFromVideos_framewise, tqdm([(tracks, videoName, srcVidDir, outRootDir, cow_ids, filterTrackPts, filterFnName) for videoName in trackInfo_frameWise.keys()], total=len(trackInfo_frameWise), desc=f"Processing videos.", unit="video"))
        # TQDM Progress bar with starmap https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm

    # Process and save the results
    finalImageCounts = {}
    for imageCount_perVideo in results:
        for cowID, count in imageCount_perVideo.items():
            if cowID not in finalImageCounts:
                finalImageCounts[cowID] = count
            else:
                finalImageCounts[cowID] += count
    # Save the results to a file
    finalCounts = [{'cowID': cowID, 'nImages': count} for cowID, count in finalImageCounts.items()]
    results_df = pd.DataFrame(finalCounts)
    results_df.to_csv(f"{outRootDir}/crops_dataset_summary{'_filtered' if filterTrackPts else '_unfiltered'}.csv", index=False)

    print(f"Processed {len(trackInfo_frameWise)} videos from. Cropped images saved to {outRootDir}.")
 


def createCropsDataset_fromArgs(args):
    tracks = pickle.load(open(args.tracksFilePath, 'rb'))

    if args.trackWise:
        create_cropsDataset_trackWise(tracks, args.srcVidDir, filterTrackPts=args.filterTrackPts, filterFnName=args.filterFnName)
    elif args.frameWise:
        create_cropsDataset_frameWise(tracks, args.srcVidDir, filterTrackPts=args.filterTrackPts, filterFnName=args.filterFnName)
    else:
        print("Error: Either --trackWise or --frameWise must be specified.")
        sys.exit(1)
    

def parseArgs():
    parser = argparse.ArgumentParser(description='Create cropped images dataset from tracks.')

    processingTypeGrp = parser.add_mutually_exclusive_group(required=True)
    processingTypeGrp.add_argument('--trackWise', action='store_true', help='Create dataset with track-wise iteration. Parallelizable, fast, but not accurate with H.264 type of encoding that do not support accurate seeking.')
    processingTypeGrp.add_argument('--frameWise', action='store_true', help='Create dataset with frame-wise iteration. Slower, but more accurate with H.264 type of encoding that do not support accurate seeking.')

    parser.add_argument('-s', '--srcVidDir', type=str, required=True, help='Source directory containing the video files.')
    parser.add_argument('-t', '--tracksFilePath', type=str, required=True, help='Path to the tracks pickle file. GT Labels must be attached to the tracks.')
    parser.add_argument('-o', '--outRootDir', type=str, required=True, help='Output directory to save the cropped images.')
    parser.add_argument('-f', '--filterTrackPts', action='store_true', help='Filter track points based on the filter function name provided.')
    parser.add_argument('-n', '--filterFnName', type=str, default='trackPtsFilter_byProximityOfRBboxToFrameCenter', help='Name of the filter function to filter track points. Default is "proximityOfRBboxToFrameCenter".')

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    createCropsDataset_fromArgs(args)