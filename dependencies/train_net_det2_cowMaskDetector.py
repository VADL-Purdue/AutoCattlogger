#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Author: Manu Ramesh
Modified from detectron2's train_net.py

A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import pdb, numpy as np

import sys
sys.path.insert(0, './detectron2/')  # Manu: to ensure that local detectron2 is imported

import detectron2.utils.comm as comm
from   detectron2.checkpoint import DetectionCheckpointer
from   detectron2.config import get_cfg
from   detectron2.data import MetadataCatalog
from   detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

#original
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
#Manu
#from detectron2.evaluation import  CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, SemSegEvaluator, verify_results

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances

##################################################################################################
##################### REGISTER CUSTOM DATASETS HERE ##############################################


#For Orbbec Camera data (also has images from OpenBarn2024 Data)
register_coco_instances("cow_topView_dataset_v6_train", {}, "../data/kp_dataset_v6/annotations/kp_dataset_v6_train.json", "../data/kp_dataset_v6/images/images_train")
register_coco_instances("cow_topView_dataset_v6_test", {}, "../data/kp_dataset_v6/annotations/kp_dataset_v6_test.json", "../data/kp_dataset_v6/images/images_test")


##################################################################################################
###################### SET METADATA CATALOG ENTRIES HERE #########################################
# This is needed only if you train keypointRCNN keypoint detection models. But we have moved to using HRNet models for keypoint detection from MMPOSE library.
# You need not touch this portion of the code unless you decide to add more keypoints.

from detectron2.data import MetadataCatalog
#detectron2 bug needs us to specify these MetadataCatalog entries
#https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#metadata-for-datasets
#https://github.com/facebookresearch/detectron2/issues/205#issuecomment-726052972

keypoint_names = kpn = [
        "left_shoulder",
        "withers",
        "right_shoulder",
        "center_back",
        "left_hip_bone",
        "hip_connector",
        "right_hip_bone",
        "left_pin_bone",
        "tail_head",
        "right_pin_bone"
      ]

keypoint_connection_rules = \
[(kpn[1-1], kpn[4 -1], (0, 0, 255) ),
 (kpn[1-1], kpn[5 -1], (0, 255, 0) ),
 (kpn[1-1], kpn[2 -1], (255, 0, 0) ),
 (kpn[2-1], kpn[3 -1], (255, 255, 0) ),
 (kpn[2-1], kpn[4 -1], (0, 255, 255) ),
 (kpn[3-1], kpn[4 -1], (0, 0, 0) ),
 (kpn[3-1], kpn[7 -1], (255, 255, 255) ),
 (kpn[4-1], kpn[6 -1], (255, 128, 128) ),
 (kpn[4-1], kpn[5 -1], (128, 255, 128) ),
 (kpn[4-1], kpn[7 -1], (128, 128, 255) ),
 (kpn[5-1], kpn[6 -1], (255, 255, 128) ),
 (kpn[5-1], kpn[8 -1], (255, 128, 255) ),
 (kpn[6-1], kpn[7 -1], (128, 255, 255) ),
 (kpn[6-1], kpn[8 -1], (255, 128, 64) ),
 (kpn[6-1], kpn[10-1], (255, 64, 128)  ),
 (kpn[6-1], kpn[9 -1], (128, 255, 64) ),
 (kpn[7-1], kpn[10-1], (128, 64, 255)  ),
 (kpn[8-1], kpn[9 -1], (64, 255, 128) ),
 (kpn[9-1], kpn[10-1], (64, 128, 255) )]


#setting metadata catalog - Manu
customDatasetsList =  []
customDatasetsList += ["cow_topView_dataset_v5_train", "cow_topView_dataset_v5_test"]
customDatasetsList += ["cow_topView_dataset_v6_train", "cow_topView_dataset_v6_test"]

for customDataset in customDatasetsList:
    MetadataCatalog.get(customDataset).keypoint_names = keypoint_names
    MetadataCatalog.get(customDataset).keypoint_flip_map = [("left_shoulder", "right_shoulder"), ("left_hip_bone", "right_hip_bone"), ("left_pin_bone", "right_pin_bone")]
    MetadataCatalog.get(customDataset).classes = ['cow_tv_mask']
    MetadataCatalog.get(customDataset).keypoint_connection_rules = keypoint_connection_rules


###################### END OF CUSTOM DATASET REGISTRATION #########################################
###################################################################################################

###################################################################################################
####################### DEFINING TRANSFORMS FOR TRAINING DATA LOADER ##############################
#by Manu
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:

        print(f"MANU: Dataset name = {dataset_name}, evaluator_type = {evaluator_type}, entering coco evaluation") #dataset name = cow_topView_dataset_v2_test, evaluator type = coco
        #quit()
        #evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        #overridden by MANU - VERY VERY IMPORTANT!
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS))

    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    #by Manu
    #augmentations are listed in detectron2/detectron2/data/transforms - augmentation_impl.py
    #https://detectron2.readthedocs.io/en/latest/tutorials/augmentation.html - might not help
    #https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html - use this instead!
    #Example https://github.com/facebookresearch/detectron2/blob/main/projects/DeepLab/train_net.py
    #I am overwriting the build_train_loader function as described in the documentation page above
    #This function is actually defined in the grand parent class of this class, yes! go hunting
    #that function calls build_detection_train_loader detectron2/data/build, so, i return that here with the changed mapper
    #to return that, i need to Import it in this file, which i have rightly done above.
    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg,
                                 mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                     #T.Resize((800, 800))
                                     T.RandomFlip(prob=0.5),
                                     T.RandomRotation(angle=[0, 90, 180, 270]),
                                     T.RandomApply(T.RandomBrightness(intensity_min=0.4, intensity_max=1.4), prob=0.25),
                                     T.RandomApply(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5), prob=0.25),
                                     T.RandomApply(T.RandomLighting(scale=1.0), prob=0.25),
                                     T.RandomApply(T.RandomContrast(intensity_min=0.5, intensity_max=1.5), prob=0.25),
                                     T.RandomApply(T.RandomCrop(crop_type = 'relative_range', crop_size = (0.75, 0.75)), prob=0.25), #randomly crops from 0.75 to H and 0.75 to W
                                 ]
                                ))
        return dataloader

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # by Manu
    #cfg.defrost()
    #cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((10, 1), dtype=float).tolist()
    #cfg.freeze()

    print(f"Stage1> len of keypoint oks sigmas = {len(cfg.TEST.KEYPOINT_OKS_SIGMAS)}")
    #quit()

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:

        print(f"Stage2> len of keypoint oks sigmas = {len(cfg.TEST.KEYPOINT_OKS_SIGMAS)}")
        #quit()

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


#by Manu
#wrapper function
# def launchCowModelTraining(datasetsDict=None, args=None):
#     '''

#     :param datasetsDict: Dictionary of datasets for registration in format: {"datasetName" : {"annJsonPath": <path-to-annotation-json-file>, "imgDir": <path-to-images-directory>}, }
#     :param args: Arugments for launch and main functions. launch() will access some args, remaining will be passed to main function.
#                  These args include which config file to use for training.
#     :return:
#     '''

#     if datasetsDict is not None:

#         for datasetName, datasetDetails in datasetsDict.items():

#             # register the dataset first
#             #example
#             #register_coco_instances("cow_topView_dataset_SEP5-fromSEP3+_R50FPN_trainSize0.25_maxHard-filter1exp_train", {}, "/home/manu/custom_cache/Datasets/SURABHI_Datasets/SURABHI_KeypointsDatasets/SEP5_Datasets/SEP5_fromSEP3+/R50FPN_trainSize_0.25_maxHard-filter1/R50FPN_trainSize_0.25_maxHard-filter1_expDataset.json", "/home/manu/custom_cache/Datasets/SURABHI_Datasets/SURABHI_KeypointsDatasets/SEP5_Datasets/SEP5_fromSEP3+/R50FPN_trainSize_0.25_maxHard-filter1/R50FPN_trainSize_0.25_maxHard-filter1_expDataset_images")
#             register_coco_instances(datasetName, {}, datasetDetails['annJsonPath'], datasetDetails['imgDir'])

#             #register metadata
#             #keypoint names and connection rules are defined as global variables above
#             MetadataCatalog.get(datasetName).keypoint_names = keypoint_names
#             MetadataCatalog.get(datasetName).keypoint_flip_map = [("left_shoulder", "right_shoulder"), ("left_hip_bone", "right_hip_bone"), ("left_pin_bone", "right_pin_bone")]
#             MetadataCatalog.get(datasetName).classes = ['cow_tv_mask']
#             MetadataCatalog.get(datasetName).keypoint_connection_rules = keypoint_connection_rules

#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
