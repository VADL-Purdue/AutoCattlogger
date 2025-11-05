#Author Manu Ramesh

dataset_info = dict(
    dataset_name='cow_tv_keypoints_train',
    #dataset_name='CowTvKeypointsDataset2',
    paper_info=dict(
        author='Manu Ramesh',
        title='Cow Top View Dataset Train Summer 22 Day 1',
        container='Purdue Dairy Datasets',
        year='2022',
        homepage='https://engineering.purdue.edu/VADL/',
    ),



    keypoint_info={
        0: dict(name='left_shoulder',   id=0, color=[  0, 100,   0], type='upper', swap='right_shoulder'),
        1: dict(name='withers',         id=1, color=[  0,   0, 139], type='upper', swap=''),
        2: dict(name='right_shoulder',  id=2, color=[176,  48,  96], type='upper', swap='left_shoulder'),
        3: dict(name='center_back',     id=3, color=[255,  69,   0], type='upper', swap=''),
        4: dict(name='left_hip_bone',    id=4, color=[255, 215,   0], type='lower', swap='right_hip_bone'),
        5: dict(name='hip_connector',   id=5, color=[  0, 255,   0], type='lower', swap=''),
        6: dict(name='right_hip_bone',  id=6, color=[  0, 255, 255], type='lower', swap='left_hip_bone'),
        7: dict(name='left_pin_bone',   id=7, color=[255,   0, 255], type='lower', swap='right_pin_bone'),
        8: dict(name='tail_head',       id=8, color=[100, 149, 237], type='lower', swap=''),
        9: dict(name='right_pin_bone',  id=9, color=[255, 218, 185], type='lower', swap='left_pin_bone'),
    },

    # From detectron2 configs
    # [0    (kpn[1-1], kpn[4 -1], () ),
    # 1    (kpn[1-1], kpn[5 -1], () ),
    # 2    (kpn[1-1], kpn[2 -1], () ),
    # 3    (kpn[2-1], kpn[3 -1], () ),
    # 4    (kpn[2-1], kpn[4 -1], () ),
    # 5    (kpn[3-1], kpn[4 -1], () ),
    # 6    (kpn[3-1], kpn[7 -1], () ),
    # 7    (kpn[4-1], kpn[6 -1], () ),
    # 8    (kpn[4-1], kpn[5 -1], () ),
    # 9    (kpn[4-1], kpn[7 -1], () ),
    # 10   (kpn[5-1], kpn[6 -1], () ),
    # 11   (kpn[5-1], kpn[8 -1], () ),
    # 12   (kpn[6-1], kpn[7 -1], () ),
    # 13   (kpn[6-1], kpn[8 -1], () ),
    # 14   (kpn[6-1], kpn[10-1], ()  ),
    # 15   (kpn[6-1], kpn[9 -1], () ),
    # 16   (kpn[7-1], kpn[10-1], ()  ),
    # 17   (kpn[8-1], kpn[9 -1], () ),
    # 18   (kpn[9-1], kpn[10-1], () )]

    skeleton_info={
        0   : dict(link=('left_shoulder', 'center_back'), id=0 , color=[0, 0, 255]),
        1   : dict(link=('left_shoulder', 'left_hip_bone'), id=1 , color=[0, 255, 0]),
        2   : dict(link=('left_shoulder', 'withers'), id=2 , color=[255, 0, 0]),
        3   : dict(link=('withers', 'right_shoulder'), id=3 , color=[255, 255, 0]),
        4   : dict(link=('withers', 'center_back'), id=4 , color=[0, 255, 255]),
        5   : dict(link=('right_shoulder', 'center_back'), id=5 , color=[0, 0, 0]),
        6   : dict(link=('right_shoulder', 'right_hip_bone'), id=6 , color=[255, 255, 255]),
        7   : dict(link=('center_back', 'hip_connector'), id=7 , color=[255, 128, 128]),
        8   : dict(link=('center_back', 'left_hip_bone'), id=8 , color=[128, 255, 128]),
        9   : dict(link=('center_back', 'right_hip_bone'), id=9 , color=[128, 128, 255]),
        10  : dict(link=('left_hip_bone', 'hip_connector'), id=10, color=[255, 255, 128]),
        11  : dict(link=('left_hip_bone', 'left_pin_bone'), id=11, color=[255, 128, 255]),
        12  : dict(link=('hip_connector', 'right_hip_bone'), id=12, color=[128, 255, 255]),
        13  : dict(link=('hip_connector', 'left_pin_bone'), id=13, color=[255, 128, 64]),
        14  : dict(link=('hip_connector', 'right_pin_bone'), id=14, color=[255, 64, 128]),
        15  : dict(link=('hip_connector', 'tail_head'), id=15, color=[128, 255, 64]),
        16  : dict(link=('right_hip_bone', 'right_pin_bone'), id=16, color=[128, 64, 255]),
        17  : dict(link=('tail_head', 'tail_head'), id=17, color=[64, 255, 128]),
        18  : dict(link=('right_pin_bone', 'right_pin_bone'), id=19, color=[64, 128, 255]),

    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])