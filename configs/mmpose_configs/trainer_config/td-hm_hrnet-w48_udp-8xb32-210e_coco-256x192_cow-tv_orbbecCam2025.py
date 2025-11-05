'''
Modified by Manu Ramesh
For cow top view keypoints training.

'''


# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from mmpose.configs._base_.default_runtime import *

from mmengine.dataset import DefaultSampler
from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, MultiStepLR
from torch.optim import Adam

from mmpose.codecs import UDPHeatmap
from mmpose.datasets import (CocoDataset, GenerateTarget, GetBBoxCenterScale,
                             LoadImage, PackPoseInputs, RandomFlip,
                             RandomHalfBody, TopdownAffine)
from mmpose.datasets.transforms.common_transforms import RandomBBoxTransform
from mmpose.evaluation import CocoMetric
from mmpose.models import (HeatmapHead, HRNet, KeypointMSELoss,
                           PoseDataPreprocessor, TopdownPoseEstimator)

# runtime
#train_cfg.update(max_epochs=210, val_interval=10)
train_cfg.update(max_epochs=210, val_interval=50)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type=Adam,
    lr=5e-4,
))


#from Prajwal's code
#channel_cfg = dict(
#    dataset_joints=10,
#    dataset_channel=[
#        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#    ],
#    inference_channel=[
#        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#    ])


# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks.update(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type=UDPHeatmap, input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type=HRNet,
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w48-8ef0771d.pth'),
    ),
    head=dict(
        type=HeatmapHead,
        in_channels=48,
        #out_channels=17,
        out_channels=10, #Manu: Should match the number of keypoints!
        deconv_out_channels=None,
        loss=dict(type=KeypointMSELoss, use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = CocoDataset #CowTvKeypointsDataset #CowTvKeypointsDataset2 #CowTvKeypointsDataset #CocoDataset
data_mode = 'topdown' #'bottomup' #'topdown' #bottom up not compatible with our dataset
data_root = '../data/kp_dataset_v6/'  #'data/coco/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction='horizontal'),
    dict(type=RandomHalfBody),
    dict(type=RandomBBoxTransform),
    dict(type=TopdownAffine, input_size=codec['input_size'], use_udp=True),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]
val_pipeline = [
    dict(type=LoadImage, backend_args=backend_args),
    dict(type=GetBBoxCenterScale),
    dict(type=TopdownAffine, input_size=codec['input_size'], use_udp=True),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size= 6, #32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/kp_dataset_v6_train.json',  #'annotations/person_keypoints_train2017.json',
        #ann_file='annotations/kp_dataset_v6_test.json',  #'annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='images/images_train/'),  #dict(img='train2017/'),
        #data_prefix=dict(img='images/images_test/'),  #dict(img='train2017/'),
        pipeline=train_pipeline,
        # metainfo=dict(from_file='configs/_base_/datasets/cow_tv_keypoints.py'), # from https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html
        metainfo=dict(from_file='../configs/mmpose_configs/metadata_config/cow_tv_keypoints.py'), # from https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html
    ))

val_dataloader = dict(
    batch_size=6, #32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/kp_dataset_v6_test.json', #'annotations/person_keypoints_val2017.json',

        bbox_file=None, #from https://mmpose.readthedocs.io/en/latest/faq.html
        #bbox_file='COCO_val2017_detections_AP_H_56_person.json', #from https://mmpose.readthedocs.io/en/latest/faq.html
        #bbox_file = f'{data_root}/annotations/kp_dataset_v6_test.json',
        #bbox_file= 'data/coco/person_detection_results/'
        #'COCO_val2017_detections_AP_H_56_person.json', #results in Key error

        data_prefix=dict(img='images/images_test/'), #dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
        # metainfo=dict(from_file='configs/_base_/datasets/cow_tv_keypoints.py'), # from https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html
        metainfo=dict(from_file='../configs/mmpose_configs/metadata_config/cow_tv_keypoints.py'), # from https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=CocoMetric,
    #ann_file=data_root + 'annotations/person_keypoints_val2017.json'
    ann_file=data_root + 'annotations/kp_dataset_v6_test.json'
    )
test_evaluator = val_evaluator

#val_cfg = None #writing this after facing error --> ValueError: val_dataloader, val_cfg, and val_evaluator should be either all None or not None,  but got val_dataloader=None, val_cfg={}, val_evaluator=None
#test_cfg = val_cfg #writing this after facing the same error for test_cfg