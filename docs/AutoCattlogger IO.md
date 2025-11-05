# AutoCattlogger IO
Author: Manu Ramesh

Volunteers can fill in more information.

---

This file discusses the inputs and outputs of the AutoCattlogger.

## Inputs to the AutoCattlogger
- (Required) A set of top view videos of cows.
- (Optional) A csv file with list of cows in order of appearance. The column header (first row of the column) should be 'cowID', and the subsequent rows should have the list of cows in the order in which they appear under the camera. 

### Dependencies
- MaskRCNN model weights from Detectron2.
- HRNet model weights from MMPOSE.
- Cow shape model for the keypoints rules-checker.

## Outputs from the AutoCattlogger

### Before attaching the ground truth labels (cowID):
Running the [generateCattlogs.py](../generateCattlogs.py) module without providing the CSV file with ground truth information generates the following outputs in the specified output directory.

- **Cow tracks** Python dictionaries as pickle files:
    - openTracks.pkl: Has all cows that have not yet exited the scene. After all videos are processed, all tracks are closed, so this dictionary should be empty.
    - closedTracks.pkl: Has tracks of all cows that have exited the scene. 
    - requiredTracks.pkl: Has closed tracks after post-porcessing. Post-processing filters out unwanted cow tracks, eg: we filter out cows that walk in directions other than the required direction.
- **The log file:** log_autoCattlog_OnVideoSet_multiInstances_QR.log
    - Go thorugh this log file for detailed information about each cow instance. The last few lines of this log will also have the AutoCattlogger statistics over all the processed videos.
- **The cattlog:** Python dictionary as a pickle file (cowDataMatDict_autoCattlogMultiCow_blk16.p)
    - Python dictionary with the format: {'trackId_1':{'blk16':'0001111.....01111'}, 'trackId_2':...}, where blk16 is the block size used to pixelate the template aligned image (refer the AutoCattlogger paper for more details), the binary string represents the barcode.
    - The binary string is 2048 bits long. This represents the 32 X 64 barcode of the cow.
    - The trackId need not be consequtive as the many false positive tracks in between are automatically rejected.
- **The metrics dictionary:** Python dictionary as a pickle file (metricsDict_autocattlog_onVideoSet_QR.p)
    - Has the raw statistics from processing all videos.
    - Has the following items: 
        - 'countsDict'
            - total_seenFrames: Total number of frames seen by the AutoCattlogger
            - total_framesWithAtLeaset1bbox: Total number of frames with at least one detection
            - total_instanceBBoxes: Total number of instance bounding boxes (total detections).
            - total_AR_area_pass: Total number of the the above detections that have AR (Aspect Ratio) and Area ratio (boxArea/frameArea) in the requried ranges.
            - total_allKps_det_inter: Total instances passing the AR and area filter above, that have all 10 keypoints detected (includes interpolated keypoints).
            - total_allKPRulePass: Total number of instances that have all 10 keypoints detected/interpolated/rectified correctly. A correct set of keypoints pass all rules of the rules-checkers.
            - total_correct_preds: Total number of correct ID preditions. This will be 0 for the autoCattlogger. This field is used by AutoCattleID at the time of inference.
            - total_kpCorrected_instances: Total number of instances that successfully underwent keypoint correction.
            - gt_locs_list: Empty list here. But is used by AutoCattleID during inference. Used to produce final accuracy metrics. Can be ignored by user.
            - topK_VideoLevelDict: Empty list here. But is used by AutoCattleID during inference. Used to produce final accuracy metrics. Can be ignored by user.
            - nVideos: Total number of videos processed.
            - cowsWithNoCorrectPreds_list: List of cows with no correct predictions. Empty here. Used by AutoCattleID. 
        - kpRulesPassCounts: (array) Histogram of number of keypoint rules passed (of the 21 rules of rules-checker1). Can be ignored by the user.
        - nKPDetectionCounts: (array) Histogram of number of keypoints detected per image (from 0 to 10).
- **Output videos:**
    - These are avi video files with all the AutoCattlogger output frames. Example frames are shown in the [main readme file](../README.md).
    - These video files do not contain frames without any detections in them.

- **Sample cropped images** Directory with .jpg images whose names are in the format 'trackId_\<trackId\>'.
    - *autoCattlog_sampleCrops_fromAllClosedTracks:* A directory with sample images for each closed track. The first instnace of each cow with correctly detected keypoints is cropped, aligned to the standard template and saved as a sample image.
    - *autoCattlog_sampleCrops_fromRequiredTracks:* Same as above, but for cows from required tracks only.

### After attaching the ground truth labels (cowID):
Once you run the [generateCattlogs.py](../generateCattlogs.py) module again by providing the ground truth csv file, it generates the following additional files. 
Note that, for programatic convenience, these are generated by separate functions, and not by the base AutoCattlogger. 

- **Cow tracks** Python dictionary as a pickle file
    - tracks_withGTLabels.pkl: Same tracks as above, but each track dictionary now has a 'gt_label' field with the cowID string as its value.

- **The cattlog:** Python dictionary as a pickle file (cowDataMatDict_autoCattlogV2_withGTLabels.p)
    - The same cattlog as above, but with the cowIDs as keys instead of track IDs. Cow IDs are usually 4 digit numbers saved as strings.
    - This dictionary has the format: {'cowID1': {'blk16':'0001111.....01111'}, 'cowID2':...}.

- **Sample cropped images** Directory
    - autoCattlogV2_sampleCrops_withGTLabels: Sample crop images with names containing the cowIDs instead of the trackIDs.

- **Sample barcodes** Directory 
    - pixBinImages_withGTLabels: Pixelated binary images (barcodes) with ground truth labels attached. For ease of interpretability, we convert the bit-vectors in the cattlog to actual barcode images and save them as .jpg files in this folder. Each image is saved in the format '\<cowID\>_blk\<block size\>'. Here, block size represents the block size used to pixelate the canonical instances of cows to generate the barcodes (refer the AutoCattlogger 2025 paper). 
    - Note that these pixelated binary images need not have a one-to-one correspondance with the sample cropped images. This is because sample cropped images are, as the name suggests, just a sample - the first good image from the track. But, the barcodes are averaged from multiple, filtlered instances from the track.
    
- **Cuts information:** CSV file with cuts information.
    - A CSV file with cuts information - has the start and end frame numbers for each required cow track.
    - You can use this file to cut videos from the top view camera or any synchronized cameras (with the required offset) to generate video segments. These segments can be used to train other cattle video analytics systems.
    
 
