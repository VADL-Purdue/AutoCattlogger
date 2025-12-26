# Models for AutoCattlogger

Author: Manu Ramesh

---

This folder should have the weights for models used by subsystems of the AutoCattlogger (and AutoCattleID), and also have the illumination maps for color correction.

- detectron2_models: for maskRCNN
- mmpose_models: for HRNet keypoint detector
- illuminationModels: has the scene illumination maps

For instructions on training/obtaining maskRCNN weights and HRNet weights, refer the [dependencies readme file](../dependencies/readme.md).

For instructions on generating the scene illumination maps, follow instructions in the [BMLP readme file](../autoCattlogger/bmlp/readme.md).

## Downloading our pretrained model weights

If you choose to download our weights, you can skip the training.

### MaskRCNN model weights
  - To download our weights on a desktop with GUI web browser, [click here](https://app.box.com/s/aa8cu8cv0pszhj64446yiim9dtpkjs6h).
    - Create a folder called 'detectron2_models' in [this very folder](./).
    - After that, Copy our weights into the [detectron2_models folder](./detectron2_models/).
  - To download it using cli, run this command
      ```
      mkdir detectron2_models #create a folder for saving the weights
      curl -L  https://app.box.com/shared/static/aa8cu8cv0pszhj64446yiim9dtpkjs6h --output ./detectron2_models/maskRCNN_kpDatasetv6_orbbec2025_model_final.pth
      ```

### HRNet keypoint detector model weights
  - To download our weights on a desktop with a GUI web browser, [click here](https://app.box.com/shared/static/0lh7opyprerc10rhw660bealryyltumq.pth).
    - Create a folder called mmpose_models in [this very folder](./).
    - Copy our weights into the [mmpose_models folder](../models/mmpose_models/).
  - To download it using cli, run this command
      ```
      mkdir ../models/mmpose_models #create a folder for saving the weights
      curl -L https://app.box.com/shared/static/0lh7opyprerc10rhw660bealryyltumq.pth --output ./mmpose_models/epoch_210_kpDatasetV5_2024.pth
      ```