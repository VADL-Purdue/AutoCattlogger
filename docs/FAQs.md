# FAQs
Author: Manu Ramesh

Volunteers can fill in more information.

---
### Why do parts of the sample crop images look cut off?

We save only the first instance from the track with valid keypoints as the sample cropped image.
Parts of the cow might not have appeared in the frame yet when this happens. So, they are blacked out.

These samples are only for your reference.

The cattlog barcodes (bit-vectors) that are generated do not consider these instances (because we filter track points based on proximity to center of the frame). So you need not worry about such instances corrupting the barcodes.

### Why do I get low IoU warnings when running the BMLP illumination map generation code?

The illumination map generator takes seeks to the frame of every track-point (instance) of the selected cow-track and runs the mask detector on the frame. 
It then selects the mask corresponding to the track-point of the given cow by checking for highest overlap with the rotated bounding box of the track-point (this info is part of the track-point data).

Ideally, the map generator seeks to the correct frame and outputs the same set of masks as during cattlogging. So, the IoU should be 100%. 
However, if the soruce videos are encoded in H.264 or similar formats, the OpenCV function cannot accurately seek to the given frame number.

- But it is ok as long as no cow is near the black cow, as the entire mask of the cow with the highest IoU is considered for generating the illumination map.
- Otherwise, use data encoded in formats such as MJPG that support accurate seeking.

The masks can be slightly different if you cattlog and generate illumination maps using videos with two different encoding standards.


### What is the difference between open tracks, closed tracks and required tracks in the autoCattlogger output?

Open tracks are those tracks that are still being processed, meaning that the cow pertaining to this track has still not left the scene. 
At the end of cattlogging, all open tracks are closed. So, the saved pickle file openTracks.pkl should be empty.

Closed tracks are those tracks that have been marked completed for processing, meaning that the cow pertaining to this track has already left the scene. 
The closedTracks.pkl file should have tracks of all cows along with tracks without any cow (false positives). 
The tracks without any cow in them can easily be removed by checking if the 'hasCow' flag is False.

In some cases, not all closed tracks with cows are useful while creating the cattlog. 
For example, in our experiments discussed in the AutoCattlogger paper, all cows walk under the camera from left to right. However, sometimes cows can walk backwards or forwards from right to left. These cases are often undocumented by human annotators. Their direction of motion is difficult to control (they weigh nearly half a metric ton).
So, we provide an option to post-process the obtained tracks using post-processing functions. These functions are used to filter for tracks of cows that start and end in particular directions (left-to-right). 
The requiredTracks.pkl file will have all such tracks.
All requiered tracks have cows in them -- there are no false positives stored here.

### What are post-processing functions? What functions are currently supported?

Post-processing functions help you filter for required tracks during cattlogging. We supply functions that filter for cows that start from left and end on the right (and vice-versa). 
You can add your own post-processing functions to filter for desired tracks.

### What is track-point filtering? Why is it necessary?

While post-processing functions help you eliminate unnecessary tracks, not all track-points are required to generate high quality cow-barcodes. 
Cow instances (track-points) towards the edges of the frames tend to have parts of cows missing, or have perspective distortions. 
Track-point filtering enables us to select only those track-points that are closer to the center of the camera frame, which are more likely to be free from the above mentioned distortions, to create high quality cow-barcodes.
You can add your own track-point filters to suit your needs.

### What does V2 mean in autoCattloggerV2 or autoCattlogV2 that I see in the code files and in some file names?

V2 stands for version 2, which is an internal reference. This is the same version that was used to generate the results presented in the AutoCattlogger paper (2025). Version1 was primitive and did not include tracking. The reader can ignore these internal numbering schemes.