# AutoCattlogger dependencies
Author: Manu Ramesh
---

This folder has all the required submodules that are required by the AutoCattlogger. It also has scripts to use the submodules.

## Detectron2:
We use detectron2 to train and use the top-view cow-mask detector.

Detectron2 works inside our supplied docker container. However, if you choose to train it separately, follow the installation instructions in their git repo [here](https://github.com/facebookresearch/detectron2.git).

If you choose to download our weights, you can skip the training.
  - To download our weights on a desktop with GUI web browser, [click here](https://app.box.com/s/aa8cu8cv0pszhj64446yiim9dtpkjs6h).
  - To download it using cli, run this command
      ```
      mkdir ../models/detectron2_models #create a folder for saving the weights
      curl -L  https://app.box.com/shared/static/aa8cu8cv0pszhj64446yiim9dtpkjs6h --output ../models/detectron2_models/maskRCNN_kpDatasetv6_orbbec2025_model_final.pth
      ```
  - Copy our weights into the [detectron2_models folder](../models/detectron2_models/).


### Train the mask detector
Use the training script in this directory and the config file under the [configs directory](../configs/detectron2_configs/) to train the maskRCNN model.
-   Activate the required env (you can download the default detectron2 env from their instructions.) for training the maskRCNN model.
```
conda activate detectron2 #or det2_openmmlab if that works for you #skip this line if you use our docker container

#go to the folder with the training script
cd ..

# run the training script, set the num-gpus value depending on your system's capabilities
python train_net_det2_cowMaskDetector.py --num-gpus=1 --config-file ../configs/detectron2_configs/COW-InstanceSegmentation/mask_rcnn_R_50_C4_cow_tv.yaml

# you can also refer to the list of available options using the --help option
python train_net_det2_cowMaskDetector.py --help

# The --resume option can be particularly helpful if your training script crashed for some reason.
```

- Feel free to modify the config file to 
    - accommodate the batch on your GPU
    - change the datasets for training and validation
    - change the testing frequency

- You can check the loss values using the tesnsorboard gui. Google for instructions.
- Wait for the training to complete. Should take 20hrs. (can vary according to batch size and gpu capcaity)
- The outputs should be saved to the ../models/detectron2_models/output_mask_rcnn_R_50_FPN_cow_tv/ directory.
- We prefer using the final checkpoint - model_final.pth. Feel free to use the checkpoint of your choice based on the loss value.

**Notes:**
-  To train the maskRCNN model on your own dataset, you must also make edits to the train_net_det2_cowMaskDetector.py file in addition to the config yaml file.
- You can also train keypointRCNN models if you choose to. Supply your own config files by modifying example config files in the detectron2 repository. The train_net_det2_cowMaskDetector.py has the required meteadata to support this training. 
  - Since we have moved from keypointRCNN to HRNet (trained with MMPOSE), we are not supplying the config file to train keypointRCNN models.

### Troubleshooting
- If you face errors with setuptools, install this version. More info [here](https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version).
    ```
    python -m pip install setuptools==59.5.0
    ```
- AutoCattlogger was designed to work with commit version: f89e39a from 25 Jul 2020. You can checkout that version in case your version errors out either while training or during inference. Do it with this command:
  ```
  cd detectron2
  git checkout f89e39ab8220a2981157f5060b5c73058c163516
  ```
- If you are unable to train the model using our supplied det2_openmmlab conda environment, you could setup a separate conda environment for this purpose. Use the setup.py file under their repo to install requirements.
- CUDA Out of memory error: Make sure to set the appropriate batch size in the [detectron2 trainer config file](../configs/detectron2_configs/COW-InstanceSegmentation/mask_rcnn_R_50_C4_cow_tv.yaml) so that an entire batch fits in your GPU memory.

## MMPOSE
MMPOSE is a framework that allows us to train and use HRNet for top view keypoint detection. 
The HRNet model performs much better than KeypointRCNN.
We use the same cow keypoints dataset to train the HRNet models because MMPOSE is compatible with COCO-style annotations.

-	For training the mmpose hrnet model, you need a dataset metadata file, a trainer config file and, dataset with annotations and images.
- The following files are in the main branch of mmpose_cows repo
  - Metadata file: configs/_base_/datasets/cow_tv_keypoints.py
    - This file defines the keypoint names and their connections. These connections are used to build the keypoint skeleton.
  - Trainer config file: mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_cow-tv.py
    - This file will take as input the paths to the metadata file and the dataset (directory of training images).
    - You will need to use this file for training the keypoint detector.
- To train the model, you simply use the trainng script provided by the mmpose guys and pass the path to the trainer config file as the argument.

### Installation 

MMPOSE works in our supplied docker container. It is automatically downloaded when you download our GitHub repo with its submodules. However, if you choose to train the keypoint detector separately, you can download and install it by following the instructions below.

Install mmpose in edit mode and train the HRNET keypoint detector. Instructions to install mmpose can be found on their [github repo](https://github.com/open-mmlab/mmpose), specifically in [this file](https://mmpose.readthedocs.io/en/latest/installation.html).
- Install mmpose from source
```
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,(THIS IS IMPORTANT)
# thus any local modifications made to the code will take effect without reinstallation.
```


### Training
You can skip this step if you choose to download our weights.
  - To download our weights on a desktop with a GUI web browser, [click here](https://app.box.com/shared/static/0lh7opyprerc10rhw660bealryyltumq.pth).
  - To download it using cli, run this command
      ```
      mkdir ../models/mmpose_models #create a folder for saving the weights
      curl -L https://app.box.com/shared/static/0lh7opyprerc10rhw660bealryyltumq.pth --output ../models/mmpose_models/epoch_210_kpDatasetV5_2024.pth
      ```
  - Copy our weights into the [mmpose_models folder](../models/mmpose_models/).

Training steps:
- Start the training with the provided training script and the config file in the configs folder. An example is shown below.
  ```
  python mmpose/tools/train.py ../configs/mmpose_configs/trainer_config/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_cow-tv_orbbecCam2025.py --work-dir ../../models/mmpose_models/cow_tv_hrnet_keypoint_detector/ --resume
  ```
- Use the --help option for more information about available training options.
  ```
  python mmpose/tools/train.py --help
  ```

### Troubleshooting

- AutoCattlogger was designed to work with commit version: 5a3be94 from 11 Jan 2024. You can checkout that version in case your version errors out either while training or during inference. Do it with this command:

  ```
  cd mmpose
  git checkout 5a3be9451bdfdad2053a90dc1199e3ff1ea1a409
  ```

## OpenCV SRGB Gamma

It is crucial for the color-correction modules under [BMLP](../autoCattlogger/bmlp/) to process the video frames in linearRGB. 
For this purpose, I use the opencv_srgb_gamma repository.
This is a small repository with simple functions to convert images from sRGB to linearRGB and back. I just decided to use this repository as the functions were already there, and I do not have to rewrite the gamma-correction and gamma-uncorrection code. 
