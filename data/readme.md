# Data for AutoCattlogger & AutoCattleID
Author: Manu Ramesh

---

## Sample videos datasets

**Please cite the [AutoCattlogger paper](https://www.sciencedirect.com/science/article/pii/S2772375525007920) if you use these datasets in part or whole.**

Follow the instructions below to download our set of sample videos. Make sure you are in this (data) folder when running the commands.

We provide sample videos from two days of Summer 2022, recorded at the Purdue dairy. These sample videos are cut-videos, that is, these have only one cow in them. However, the AutoCattlogger can also function on longer videos with multiple cows (e.g.: many cows walking along the same path for hours, or many cows walking in parallel across hours).

CLI commands:
```
# download the tar files
curl -L https://app.box.com/shared/static/rih8q2jc7dtgi9yge17wfz84y15rqkez --output ./sampleVideos1.tar
curl -L https://app.box.com/shared/static/aagwjs8b42q11xc6b6l75ccgk47s8tl4 --output ./sampleVideos2.tar

# extract the tar files
tar -xvf sampleVideos1.tar
tar -xvf sampleVideos2.tar

# remove the tar files if you don't need them
rm sampleVideos1.tar sampleVideos2.tar

```

Note that in some of the above cut-videos, there are some frames with an extra cow in them. However, since those extra cows are not fully visible in the frames, they would not be affecting the AutoCattlogger's functioning.
In case you are working with your own data, be mindfull of jump cuts/discontinuities in videos as they could confuse the cow tracker.

---

## Datasets for training/testing the keypoint and mask detectors

**Please cite the [AutoCattlogger paper](https://www.sciencedirect.com/science/article/pii/S2772375525007920) if you use these datasets in part or whole.**

If you wish to train the keypoint and mask detectors on your own data, it would be best to annoate a few images from your data and add them to the datasets we provide here. 

Use the following commands to download our VADL_PurdueCowsDataset. Make sure that your shell is in this folder (data folder) before running the commands given below.
```
# download the tar files
curl -L https://app.box.com/shared/static/brsa06zpj7ot6cb53kqbqvmmzsivgjid --output ./VADL_PurdueCowsDataset_train.tar
curl -L https://app.box.com/shared/static/dc9de5i9kh8j1jcq1atliog972r93yhp --output ./VADL_PurdueCowsDataset_test.tar

# extract the tar files
tar -xvf VADL_PurdueCowsDataset_train.tar
tar -xvf VADL_PurdueCowsDataset_test.tar

# remove the tar files if you don't need them
rm VADL_PurdueCowsDataset_train.tar VADL_PurdueCowsDataset_test.tar

```

Notes:
1. There could be a few training images without cows in them - and so, will not have annotations.
2. The dataset of training images provided here is a cut-down version of the dataset used to train the models in the AutoCattlogger. You should not expect the model trained on this data to perform the same as our supplied pre-trained models.

Stats:
- Training set: 460 images (~424 annotated - remaining images might not have cows in them)
- Testing set: 34 images (34 annotated)


## Custom datasets
You can download your own datasets into this folder.