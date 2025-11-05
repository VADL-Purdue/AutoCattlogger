# Annotating masks and keypoints
Author: Manu Ramesh

Volunteers can fill in more information.

---

In case you need to annotate more images to improve detection on your own data (videos/images), you can do so using the [coco-annotator](https://github.com/jsbroks/coco-annotator).

Follow instructions in the following videos for help with annotation.
- **Video1: Setting up the coco annotator and getting started with annotations.** [link](https://www.youtube.com/watch?v=7xDOEP2Kf-4)
- **Video2: Common mistakes and how to avoid them.** [link](https://www.youtube.com/watch?v=LLXQuWY0UB4)

# Tips:
- You need not create a new keypoint category and add the set of keypoints that we use, if you already have our dataset.
    - You can import all information about our keypoints set and use them directly.
    - For this, you must first create a new dataset on the coco-annotator webpage.
    - Then, copy all images of our dataset (train/test) into the specific dataset directory under the coco-annotator folder.
    - Hitting scan and refreshing the page should show all the copied images on the webpage.
    - Next, select 'Import COCO', navigate to the annotation json file of our dataset (train/test) and select to import it.
    - Refresh the page and then select any image to view the maks and keypoint annotations on the cows.
    - Coco-annotator would now have the keypoint category 'cow-body-tv' saved.
    - You can add more of your own images into this dataset or another and directly use the saved keypoint category 'cow-body-tv' to annotate them.

- Remeber that there are 10 keypoints. 
    - The set with fewer keypoints shown in the first tutorial video is only for demonstrating the use of coco-annotator.
    - View examples from our dataset on the coco-annotator to get an idea about annotating the keypoints.

- Remember not to draw masks on the neck region of the cow.
    - The first tutorial video has masks drawn on the neck of the cow. But this is just for demonstration purposes.
    - Use the examples in our datsets to guide you. The masks in the second tutorial video are representative of our dataset.