'''
Author: Manu Ramesh

Tools to allow user to create or modify the cattlogs.
'''

import os, sys, pickle, argparse, pdb
sys.path.insert(0, '../../') #Top level dir of the repo
from autoCattleID.helpers.helper_for_trackPointFiltering import getFilterFnFromName, getFilteredCattlog

def filterAndMakeCattlog(requiredTracksListPath, filterFnName=None, saveDir='./', **kwargs):
    '''
    Creates cattlog from filtered tracks. Tracks must have GT labels (cowIDs) attached before using this tool.
    Useful if you want to save time during inference by not computing the cattlog from the tracks on every inference run.
    Useful to save the requried cattlog so that you can combine cattlogs directly.

    Remember to postprocess the tracks before filtering to remove all tracks without cows, and to remove cows that walk in undesired directions.

    :param requiredTracksListPath: paths to the tracks pickle file (required tracks with gt_labels attached)
    :param filterFnName: Name of the filter function to be used. 
                        Paper used 'trackPtsFilter_byProximityOfRBboxToFrameCenter' with inclusionPercentage_top = 0.2, frameW=1920, frameH=1080 as kwargs.
                        For more options run getFilterFnFromName(trackPtsFilterFnName=None, justListAllAvailableFns=True).
    :param saveDir: Directory to save the filtered cattlog pickle file. Default is current directory './'.
    :param **kwargs: Keyword arguments that are passed to the filter function.

    '''

    tracksList = pickle.load(open(requiredTracksListPath, 'rb'))

    # Filter the trackpoints and create a cattlog if requested
    trackPtsFilterFn = getFilterFnFromName(filterFnName) # will be None if filterFnName is None
    cattlog = getFilteredCattlog(tracksList, trackPtsFilterFn=trackPtsFilterFn, **kwargs) #will return the unfiltered cattlogDict if trackPtsFilterFn is None

    # save the cattlog
    outDictPath = saveDir + '/filtered_cattlog.p'
    pickle.dump(cattlog, open(outDictPath, 'wb'))





def combineCattlogs(cattlog1_path, cattlog2_path, overwriteCattlog1=False, saveDir=None):
    '''
    Tool to combine the list of cows from both the cattlogs. 
    Tracks must have the ground truth labels attached to tracks before the use of this tool.

    If a cow exists in both cattlogs, the resulting cattlog will have barcode from the first cattlog.

    The cattlog with gt_labels (cowIDs attached) will be in a python dict in the format {cowID1: {'blk16': bit-vector-string}, cowID2: {...}, ...}.
    If the gt_labels are not attached, the keys will be trackIds, in the format {'trackId_<trackID1>': {'blk16': bit-vector-string}, 'trackId_<trackID2>': {...}, ...}.

    :param cattlog1_path: path to cattlog 1 pickle file (eg: cowDataMatDict_autoCattlogV2_withGTLabels.p)
    :param cattlog2_path: path to cattlog 2 pickle file (eg: cowDataMatDict_autoCattlogV2_withGTLabels.p)
    :param overwriteCattlog1: If true, overwrites the cattlog1 with the combined cattlog.
    '''

    cattlog1 = pickle.load(open(cattlog1_path, 'rb'))
    cattlog2 = pickle.load(open(cattlog2_path, 'rb'))

    # confirm that gt labels are attached by confirming that no keys have 'trackId_' in their names
    assert all(not key.startswith('trackId_') for key in cattlog1.keys()), "Please attach GT labels to cattlog1 before combining."
    assert all(not key.startswith('trackId_') for key in cattlog2.keys()), "Please attach GT labels to cattlog2 before combining."

    outCattlog = cattlog2 | cattlog1 #pass cattlog 1 as second arg to prefer cattlog1 bit-vectors for common cows


    if overwriteCattlog1:
        pickle.dump(outCattlog, open(cattlog1_path, 'wb'))

    elif saveDir is not None:
        outDictPath = saveDir+'/combined_cattlog.p'
        pickle.dump(outCattlog, open(outDictPath, 'wb'))
    
    return outCattlog


def deleteCattlogEntry(cattlogPath, cowID=None, trackID=None, overwriteFile=False, saveDir=None):
    '''
    Deletes a cow from the cattlog.  
    
    :param cattlogPath: Path to cattlog pickle file. Eg: cowDataMatDict_autoCattlogV2_withGTLabels.p
    :param cowID: cowID of the cow that needs to be deleted. <str>
    :param trackID: trackID of the cow that needs to be deleted. <int>
    :param overwriteFile: If true, overwrites the original cattlog file with the modified one. <bool>
    :param saveDir: Directory to save the modified cattlog if not overwriting. <str>
    '''

    assert cowID is not None or trackID is not None, "Either cowID or trackID must be provided."

    cattlog = pickle.load(open(cattlogPath, 'rb'))

    if cowID is not None:
        if cowID in cattlog:
            del cattlog[cowID]
        else:
            print(f"CowID {cowID} not found in cattlog.")

    elif trackID is not None:
        trackKey = f'trackId_{trackID}'
        if trackKey in cattlog:
            del cattlog[trackKey]
        else:
            print(f"TrackID {trackID} not found in cattlog.")
        
    if overwriteFile:
        pickle.dump(cattlog, open(cattlogPath, 'wb'))
    elif saveDir is not None:
        outPath = saveDir + '/modified_cattlog.p'
        pickle.dump(cattlog, open(outPath, 'wb'))
    # else:
    #     pass

    return cattlog

    

def cli_parser():

    mainParser = argparse.ArgumentParser(description="Cattlog Editor CLI")

    subparsers = mainParser.add_subparsers(dest='operation', help='Type of operation to perform on cattlog')

    makeFilteredCattlogParser = subparsers.add_parser('filterAndMake', help='Filter tracks and create the cattlog based on specified criteria. Tracks must have GT labels attached.')
    makeFilteredCattlogParser.add_argument('-t', '--requiredTracksListPath', help='Path to the required tracks list pickle file', required=True)
    makeFilteredCattlogParser.add_argument('-f', '--filterFnName', default='trackPtsFilter_byProximityOfRBboxToFrameCenter', help='Name of the filter function to apply', required=False)
    makeFilteredCattlogParser.add_argument('-s', '--saveDir', default='./', required=False, help='Directory to save the filtered cattlog pickle file')


    combineParser = subparsers.add_parser('combine', help='Combine two cattlogs')
    combineParser.add_argument('-c1', '--cattlog1Path', help='Path to the first cattlog pickle file', required=True)
    combineParser.add_argument('-c2', '--cattlog2Path', help='Path to the second cattlog pickle file', required=True)
    combineParser.add_argument('-o', '--overwriteCattlog1', action='store_true', help='Overwrite the first cattlog with the combined result')
    combineParser.add_argument('-s', '--saveDir', default=None, required=False, help='Directory to save the combined cattlog if not overwriting')


    deleteCowParser = subparsers.add_parser('deleteCow', help='Delete a cow from the cattlog')
    deleteCowParser.add_argument('-c', '--cattlogPath', help='Path to the cattlog pickle file', required=True)
    
    whichCowToDelGrp = deleteCowParser.add_mutually_exclusive_group(required=True)
    whichCowToDelGrp.add_argument('--cowID', default=None, help='Cow ID to delete', required=False)
    whichCowToDelGrp.add_argument('--trackID', type=int, default=None, help='Track ID to delete', required=False)
    
    deleteCowParser.add_argument('-o', '--overwriteFile', action='store_true', help='Overwrite the original cattlog file with the modified one')
    deleteCowParser.add_argument('-s', '--saveDir', default=None, required=False, help='Directory to save the modified cattlog if not overwriting. If None, will not save.')

    args = mainParser.parse_args()

    return args

def runCattlogEditor(args):
    
    if args.operation == 'filterAndMake':
        filterAndMakeCattlog(args.requiredTracksListPath, filterFnName=args.filterFnName, saveDir=args.saveDir)
    elif args.operation == 'combine':
        combineCattlogs(args.cattlog1Path, args.cattlog2Path, overwriteCattlog1=args.overwriteCattlog1, saveDir=args.saveDir)
    elif args.operation == 'deleteCow':
        deleteCattlogEntry(args.cattlogPath, cowID=args.cowID, trackID=args.trackID, overwriteFile=args.overwriteFile, saveDir=args.saveDir)
    

if __name__ == "__main__":
    args = cli_parser()
    runCattlogEditor(args=args)
