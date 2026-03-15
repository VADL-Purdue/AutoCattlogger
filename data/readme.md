# Data for AutoCattlogger & AutoCattleID
Author: Manu Ramesh

---

## Sample videos datasets

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

## TBA

We may update this readme file with links to download sample images and annotations in the future.

## Custom datasets
You can download your own datasets into this folder.