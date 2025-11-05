# Some helpful tools for the user and instructions to use them
Author: Manu Ramesh

---

## Instructions for using the cattlog editor

The cattlog editor provides handy CLI tools to make and edit cattlogs. These tools are:
- FilterAndMake: Use this to filter the track points based on trackpoint-filter-functions and generate the cattlog from the remaining track-points. As the AutoCattlogger paper discusses, filtering trackpoints before creating the cattlog helps to generate good quality barcodes. Example usage:
    ```
    python cattlogEditor.py filterAndMake --help
    python cattlogEditor.py filterAndMake -t tracks_withGTLabels.pkl -f trackPtsFilter_byProximityOfRBboxToFrameCenter -s .
    ```
- Delete cow: Use this to delete the cattlog entry of a particular cow. You can delete it by the cowID or the trackID.
    ```
    python cattlogEditor.py deleteCow --help
    python cattlogEditor.py deleteCow --cattlogPath ./filtered_cattlog.p --cowID 6116 --saveDir .
    ```

- Combine two cattlogs with following the sample command given below. If a cow is present in both the cattlogs, the barcode in the first cattlog will be preferred. You also have an option to overwrite the first cattlog with the merged cattlog.
    ```
    python cattlogEditor.py combine --help
    python cattlogEditor.py combine -c1 cattlog1.p -c2 cattlog2.p --saveDir . 
    ```

As always, you can look at all available options with the --help flag.

## Instructions to create cow-cropped-images datasets

You can use the [makeCropsDataset](./makeCropsDataset.py) tool to generate cow cropped images. These are images of cows cropped to their oriented bounding boxes. You could use these images to train your own cattle detection/identification systems.

If you have videos with encoding schemes that support accurate seeking use the trackWise for faster processing. This option iterates through each instance in a track, seeks to the frame in the video, crops the cows and saves the cropped image.

    python makeCropsDataset.py --help
    python makeCropsDataset.py --trackWise -s <path-to-dir-with-videos> -o <output-dir> -t <path-to-tracks_withGTLabels.pkl>

If you have videos with encoding schemes such as H.264 which do not support accurate seeks, use the --frameWise option. This will iterate through every frame in the video and save crops of all cows in the frame at the same time.
Example usage:
    
    python makeCropsDataset.py --frameWise<path-to-dir-with-videos> -o <output-dir> -t <path-to-tracks_withGTLabels.pkl>

Example with filtering

    python makeCropsDataset.py --frameWise<path-to-dir-with-videos> -o <output-dir> -t <path-to-tracks_withGTLabels.pkl> -f --filterFnName trackPtsFilter_closestImgToFrameCenter

You can choose a filter among the available few trackpoint filters, or add your own.

## Instructions to create upper and lower bounds for the keypoint rules-checker(2)
We provide two rules-checkers to validate the correctness of the detected/interpolated/rectified keypoints - rulesChecker1 (RC1) and rulesChecker2 (RC2). Both these checkers check for breach of upper and/or lower bounds of certain instance statistics.
These instance statistics include - length ratios between pairs of keypoints, angle ratios between pairs of three keypoints, etc.

RC1 uses handcrafted bounds and is used for initial (and final) check. RC2 is used during keypoint-rectification and involves more rules from bounds computed from the underlying training dataset. You could choose to use only RC2 for all checks and ignore RC1 by setting the value of 'useHandCraftedKPRuleLimits' to False in the [autoCattloger config file](../../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml).

To generate the bounds for RC2, use the computeLimitsForRulesChecker.py module by following the example usage shown below.

    python computeLimitsForRulesChecker.py -h # for help
    python computeLimitsForRulesChecker.py -p <path-to-annotation.json-file>

This generates the pickle file with a python dictionary containing the upper and lower bounds for each rule of RC2. This pickle file is saved as datasetKpStatsDict_kp_dataset_v6_test.p in the \<desired output directory\>/statOutputs/ directory.
Set the path to this pickle file in the  [autoCattloger config file](../../configs/autoCattlogger_configs/autoCattlogger_example_config.yaml) to make RC2 use these bounds.

We recommend that you generate new bounds and use them regardless of you setting the 'useHandCraftedKPRuleLimits' setting to True or False. This is because RC2 is always used during keypoint rectification, and the AutoCattlogger will be more accurate if it knows the permissible range of deviation in keypoint locations.

## Instructions to use the cow-matcher GUI tool

The outputs from the AutoCattlogger is made more interpretable by attaching the ground truth labels to tracks and barcodes. To do this, the [generateCattlogs.py](../../generateCattlogs.py) script requires a ground truth CSV file with a column containing the cowIDs in the same order in which the cows appear in the scene.

If you do not have this order information, or if you need to correct errors in the order, the cow-matcher GUI tool could be very useful. Please read the [readme file]() in the cow-matcher directory for more information about how to use it.

You can use the cow matcher to generate an order-correct list of cowIDs in a CSV file and then feed the CSV file to the [generateCattlogs.py](../../generateCattlogs.py) script with the -g option to attach the ground truth labels (cowIDs) to the tracks, sample-cropped-images, and the cattlogs.

