'''
Author: Manu Ramesh | VADL | Purdue ECE
This file has code to morph cow images, given the detected keypoints, to the keypoint-template.
'''

import numpy as np
import cv2, pdb, os, glob, sys


def morph_cow_to_template(cowCropImg, warpedKPs, keypoint_names, templateKPs=None, selectedTriIds=[], printDebugInfoToScreen=False):
    '''
    Given a cropped and warped cow image and the warped keypoints, this function morphs the cow into the template cow,
    using techniques popular for face warping.

    MAKE SURE TO PASS IN ONLY THE VISIBLE KEYPOINTS. (ALTERNATIVELY, YOU COULD PASS ALL KEYPOINTS ONLY IF ALL ARE VISIBLE)

    We need to resize the images to a standard size.

    :param cowCropImg:
    :param warpedKPs: [[x,y,visibility/prob]]
    :param templateKPs:
    :param selectedTriIds: list of ids of triangles you need to morph, look at diagram in get_triangles() for id numbers. if left empty, all triangles are morphed
    :return:
    '''

    #define triangle connections for input KPs
    #define target KP locations - use dictionaries
    #define triangle connections for target KPS
    #Affine warp each triangle

    #remove the probability (visibility) values
    #if warpedKPs.shape[-1] == 3:
    tgt_KPs = np.zeros_like(warpedKPs) #initialize
    KP_prbos  = warpedKPs[:, -1]
    warpedKPs = warpedKPs[:,:-1]


    source_KP_locs = {k:v for k,v in zip(keypoint_names, warpedKPs)} #dictionary

    if len(cowCropImg.shape) == 2: #for gray images and depth images with single channel
        src_h, src_w = cowCropImg.shape
    else:
        src_h, src_w, _ = cowCropImg.shape

    tgt_h, tgt_w =    512, 256

    #the locs must be in (x, y) format
    #target_KP_locs = tkps ={"left_shoulder": (95, 0), "withers": (127, 0), "right_shoulder": (159, 0), "center_back":(127, 255), "left_hip_bone":(0, 348), "hip_connector":(127, 348), "right_hip_bone":(255, 348), "left_pin_bone":(95, 511), "tail_head":(127, 511), "right_pin_bone":(159, 511)} #original
    target_KP_locs = tkps ={"left_shoulder": (95, 0), "withers": (127, 0), "right_shoulder": (159, 0), "center_back":(127, 127), "left_hip_bone":(0, 348), "hip_connector":(127, 348), "right_hip_bone":(255, 348), "left_pin_bone":(95, 511), "tail_head":(127, 511), "right_pin_bone":(159, 511)} #has different center_back

    #ORIENTATION - 180-DEGREE ROTATION WHEN NECESSARY
    #the infer function can sometimes supply the crop images upside down
    #the cow must be oriented such that the withers are on top and the tail head is at the bottom. To do that,
    #if withers are closer to the bottom of the image than the top, we must actually rotate the image by 180degrees
    #doing so would mean that we would need change the source keypoint locations
    # we could also just change the additional keypoints instead of rotating images and also changing all the source keypoints - choosing not to do this might be difficult to understand while revisiting this section of the code
    if abs(source_KP_locs['withers'][-1] - src_h) < abs(source_KP_locs['withers'][-1] - 0):
        cowCropImg = cv2.rotate(cowCropImg, cv2.ROTATE_180)
        source_KP_locs_temp = {}
        for k, val in source_KP_locs.items():
            x, y = val
            source_KP_locs_temp[k] = (src_w-x, src_h-y)
        source_KP_locs = source_KP_locs_temp

    #these additional points must map to the same points after transformation
    additional_src_pt_locs = {'top_left_corner':(0, 0), 'top_right_corner':(src_w-1, 0), 'bot_left_corner':(0, src_h-1), 'bot_right_corner': (src_w-1, src_h-1), 'left_edge_center':(0, src_h//2-1), 'right_edge_center':(src_w-1, src_h//2-1)}
    additional_tgt_pt_locs = {'top_left_corner':(0, 0), 'top_right_corner':(tgt_w-1, 0), 'bot_left_corner':(0, tgt_h-1), 'bot_right_corner': (tgt_w-1, tgt_h-1), 'left_edge_center':(0, tgt_h//2-1), 'right_edge_center':(tgt_w-1, tgt_h//2-1)}
    #additional_pt_list = list(additional_pt_locs.values())

    #source_pts = [(x,y) for x,y in warpedKPs] + additional_pt_list
    #target_pts = list(target_KP_locs.values()) + additional_pt_list

    src_triangles = get_triangles(source_KP_locs, additional_src_pt_locs)
    tgt_triangles = get_triangles(target_KP_locs, additional_tgt_pt_locs)

    if len(cowCropImg.shape) == 2: #for gray images and depth images
        tgt_img = np.zeros((tgt_h, tgt_w), cowCropImg.dtype) #np.uint8, or np.uint16 for depth images
    else:
        tgt_img = np.zeros((tgt_h, tgt_w, 3), cowCropImg.dtype) #np.uint8

    #select only the required triangles for morphing
    selectedTriangles = []
    if selectedTriIds == []:
        selectedTriangles = list(tgt_triangles.items())
    else:
        selectedTriangles = [list(tgt_triangles.items())[triId-1] for triId in selectedTriIds] #-1 in triId-1 is because triId starts from 1 and list index starts from 0

    #for tri, vals in tgt_triangles.items():
    for tri, vals in selectedTriangles: #to select only a sequence of kps

        src_pts = np.float32(src_triangles[tri]) #src_pt0, src_pt1, src_pt2 =
        tgt_pts = np.float32(vals) #tgt_pt0, tgt_pt1, tgt_pt2 =

        if printDebugInfoToScreen: print(f"src_pts = {src_pts}\ntgt_pts = {tgt_pts}")

        M = cv2.getAffineTransform(src_pts, tgt_pts)
        if printDebugInfoToScreen: print(f"M = {M}")

        #the triangles can be morphed individually by masking each of them. Do this if you want to optimize further.
        outTri_w = np.max(tgt_pts[:,0]) - np.min(tgt_pts[:,0])
        outTri_h = np.max(tgt_pts[:,1]) - np.min(tgt_pts[:,1])

        #outTriangle = cv2.warpAffine(cowCropImg, M, (outTri_w, outTri_h))
        outTriangle = cv2.warpAffine(cowCropImg, M, (tgt_w, tgt_h))

        tri_mask = np.zeros((tgt_h, tgt_w), np.uint8)
        cv2.fillConvexPoly(tri_mask, tgt_pts.astype(int), 255) #if this doesn't work, try fillConvexPoly
        outTriangle = cv2.bitwise_and(outTriangle, outTriangle, mask=tri_mask)

        inv_tri_mask = cv2.bitwise_not(tri_mask)
        # if you do not use inv_tri_mask, you will see artifacts between triangles
        #the artifacts are because of the edge pixels of triangles getting added to the tgt_img
        # there will be an overlap of 1 pixel
        #no more median blurring is required because we are not seeing artifacts between triangles anymore
        tgt_img = cv2.bitwise_and(tgt_img, tgt_img, mask=inv_tri_mask) #this should remove one line of pixels on the common edge between the accumulated image and the triangle about to be added
        tgt_img = cv2.add(tgt_img, outTriangle)

        #cv2.imshow('outTriangle', outTriangle)
        #cv2.imshow('tri_mask', tri_mask)
        #cv2.imshow('tgt_img', tgt_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #cv2.imshow('tgt_img', tgt_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    tgt_KPs[:,:-1] = np.array(list(target_KP_locs.values()))
    tgt_KPs[:,-1]  = KP_prbos

    return tgt_img, tgt_KPs





def get_triangles(kpts, apts):
    '''

    :param kpts: keypoints dict
    :param apts: additional points dict
    :return:
    '''
    #define triangle correspondences
    #use an image for reference

    #delete later
    #kpts = {"left_shoulder": (95, 0), "withers": (127, 0), "right_shoulder": (159, 0), "center_back": (127, 255),
    # "left_hip_bone": (0, 348),
    # "hip_connector": (127, 348), "right_hip_bone": (255, 348), "left_pin_bone": (95, 511), "tail_head": (127, 511),
    # "right_pin_bone": (159, 511)}
    #apts = {'top_left_corner': (0, 0), 'top_right_corner': (tgt_w - 1, 0), 'bot_left_corner': (0, tgt_h - 1),
    #        'bot_right_corner': (tgt_w - 1, tgt_h - 1), 'left_edge_center': (0, tgt_h // 2 - 1),
    #        'right_edge_center': (tgt_w - 1, tgt_h // 2 - 1)}


    '''
    The 16 triangles that need to be transformed!
    
          This way top!
    A------B----C----D------E   Legend:
    |  1  /|    |    |\  2  |   A: top left corner
    |    / \ 3  |  4 / \    |   B: left shoulder
    |   /   \   |   /   \   |   C: withers
    |  /     \  |  /     \  |   D: right shoulder
    | /  5    \ | /   6   \ |   E: top right corner
    |/         \|/         \|   F: left edge center
    F-----------G-----------H   G: center back
    |    7     /|\    8     |   H: right edge center
    | --------- | --------- |   I: left hip bone
    |/   9      |     10   \|   J: hip connector
    I-----------J-----------K   K: right hip bone
    |\         /|\         /|   L: bot left corner
    | \   11 /  |  \  12  / |   M: left pin bone
    |  \    /   |   \    /  |   N: tail head
    |15 \  / 13 | 14 \  / 16|   O: right pin bone
    L----M------N------O----p   P: bot right corner
    '''

    #follow order while entering points, left to right, top to bottom
    helperDict = hd = {'A': apts['top_left_corner'], 'B': kpts["left_shoulder"], 'C':kpts["withers"], 'D': kpts["right_shoulder"], 'E':apts['top_right_corner'],
                  'F': apts['left_edge_center'], 'G': kpts["center_back"], 'H': apts['right_edge_center'],
                  'I': kpts["left_hip_bone"], 'J': kpts["hip_connector"], 'K':kpts["right_hip_bone"],
                  'L': apts['bot_left_corner'], 'M': kpts["left_pin_bone"], 'N': kpts["tail_head"], 'O': kpts["right_pin_bone"], 'P': apts['bot_right_corner']
                  }


    triangles = {1: (hd['A'], hd['B'], hd['F']),
                     2: (hd['D'], hd['E'], hd['H']),
                     3: (hd['B'], hd['C'], hd['G']),
                     4: (hd['C'], hd['D'], hd['G']),
                     5: (hd['B'], hd['F'], hd['G']),
                     6: (hd['D'], hd['G'], hd['H']),
                     7: (hd['F'], hd['G'], hd['I']),
                     8: (hd['G'], hd['H'], hd['K']),
                     9: (hd['G'], hd['I'], hd['J']),
                    10: (hd['G'], hd['J'], hd['K']),
                    11: (hd['I'], hd['J'], hd['M']),
                    12: (hd['J'], hd['K'], hd['O']),
                    13: (hd['J'], hd['M'], hd['N']),
                    14: (hd['J'], hd['N'], hd['O']),
                    15: (hd['I'], hd['L'], hd['M']),
                    16: (hd['K'], hd['O'], hd['P'])
                    }

    #if printDebugInfoToScreen: print(f"Triangles = {triangles}")

    return triangles


if __name__ == "__main__":
    #cowCropImg = cv2.imread("../Experiments/images/5700_256x512.jpg")
    cowCropImg = cv2.imread("../Experiments/images/5700_256x512_MARKED.jpg")
    warpedKPs = np.array([[138,2,2], [196,20,2], [245, 31,2], [139,245,2], [36,348,2], [142,358,2], [254,313,2], [121,505,2], [153,499,2], [198,499,2]])
    keypoit_names =  ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone",
     "left_pin_bone", "tail_head", "right_pin_bone"]
    #morph_cow_to_template(cowCropImg=cowCropImg, warpedKPs=warpedKPs, keypoint_names=keypoit_names)
    morph_cow_to_template(cowCropImg=cowCropImg, warpedKPs=warpedKPs, keypoint_names=keypoit_names, selectedTriIds=[3,4,5,6,7,8,9,10,11,12,13,14])