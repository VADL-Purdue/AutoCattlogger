'''
Author:  Manu Ramesh, VADL, Purdue ECE
This file is supposed to contain functions that help in interpolating missing keypoints.
I could also add functions for kp detection error correction later, if feasible.

REFACTORING INFO:
This was originally called helper_for_kpCorrection.py (module was called helper_for_kpCorrection) - this name was used in EI2023 backups.
Beware of this fact.
This will be called helper_for_kpInterpolation(.py) from now on.
There is another module that is written to deal with keypoint correction, and we will be using the helper_for_kpCorrection name for this new module introduced with SURABHI (experiment_getAffectedKPs).
'''

import numpy as np, os, cv2, matplotlib.pyplot as plt
import logging, inspect
import pdb, sys
from pycocotools.coco import COCO

sys.path.insert(0, '../../') #top level dir
from autoCattlogger.helpers.helper_for_infer import get_visible_KPs

def interpolateKPs(instance_KPs, keypoint_names, keypoint_threshold=0.5, frameW=1920, frameH=1080, printDebugInfoToScreen=False):
    '''
    Interpolates missing keypoints to max possible extent.
    :param instance_KPs: The original KPs in the frame coordinates (not the warpped ones).
    :param keypoint_names: list with names of keypoints in order
    :param keypoint_threshold: keypoint visibility lower threshold
    :param frameW: frame width. Used to check if interpolated KP is out of frame, i.e., if it is actually invisible.
    :param frameH: frame height. Used to check if interpolated KP is out of frame, i.e., if it is actually invisible.
    :return: Keypoints array with the warped kps. With standard visiblity values.
    '''

    visible_KPs, n_visibleKPs, totalKPs = get_visible_KPs(keypoints=instance_KPs, keypoint_names=keypoint_names, keypoint_threshold=0.5)

    n_invisibleKPs = totalKPs - n_visibleKPs
    if printDebugInfoToScreen: print(f"Total invisible/missing KPs = {n_invisibleKPs}")
    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Total invisible/missing KPs: {n_invisibleKPs}")

    #if printDebugInfoToScreen: print(f"\nvisible KPs = {visible_KPs}")

    prev_n_visibleKPs = n_visibleKPs #10

    #READ THIS
    #we need two loops! we cannot just iterate over invisible KPs one at a time
    #this is because in the initial loop, a kp might not have all the its required KPs for interpolation
    #these required KPs could be predicted in the initial iterations and then they could be used to predict the first KP

    while True: #do while loop

        for kpName in keypoint_names:
            #if printDebugInfoToScreen: print(f"keypoint name = {kpName}")

            if kpName not in visible_KPs: #missing KP
                if printDebugInfoToScreen: print(f"Missing KP: {kpName}")
                logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Missing KP: {kpName}")

                #for reference
                #keypoint_names = kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone", "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]

                #missing pt: hip-connector,
                if kpName == 'hip_connector':
                    # req pts: two hip bones (presence of one other point improves prediction)
                    if "left_hip_bone" in visible_KPs and "right_hip_bone" in visible_KPs and len(visible_KPs):
                        if printDebugInfoToScreen: print(f"Fixing Hip Connector")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing hip connector.")

                        #lhip (x1,y1) --------------p0------------------rhip (x2,y2)
                        #p0(x0,y0) = mid point of line segment joining two hip bones
                        #logic - interpolated hip connector is at deviationFactor times length of line segment connecting two hip bones distance away from the mid point of line segment joing two hip bones, towards the tail head or away from the withers

                        x1, y1 = visible_KPs['left_hip_bone']
                        x2, y2 = visible_KPs['right_hip_bone']
                        x0 = (x1+x2)/2; y0 = (y1+y2)/2
                        #if printDebugInfoToScreen: print(f"P0(x0,y0) = ({x0,y0})")

                        m12 = (y2-y1)/(x2-x1) if (x2-x1) !=0 else  (y2-y1)/(x2-x1+1e-10) #slope of hip bones line
                        m0 = -1/m12 if m12 != 0 else -1/(m12+1e-10)  #slope of perpendicular line, addition of small value to prevent div by 0 error
                        #if printDebugInfoToScreen: print(f"Slope m12 = {m12}, m0 = {m0}")

                        l12 = np.sqrt((y2-y1)**2 + (x2-x1)**2) #length of line segment connecting hip bones
                        deviationFactor = 0.1 #fraction of l12 that the hip connector should deviate from p0(x0,y0)
                        lhc0 = deviationFactor * l12 #length of line connecting hip bone and mid point p0

                        #if printDebugInfoToScreen: print(f"l12 = {l12}, lhc0 = {lhc0}")

                        #(xhc, yhc) = hip connector point coordinates
                        #there will be two possible points for xhc, one on each side of line joining hip bones, solve the quadratic equation
                        x_pw2_coeff = 1 + m0**2
                        x_pw1_coeff =  -2*x0 -2*m0**2 * x0 #-2*x0 + 2 * (y0 - m0 *x0) * m0 - 2*y0*m0
                        x_pw0_coeff = x0**2 * (1+m0**2) - lhc0**2 #simplified #(y0-m0*x0)**2 - 2*y0*(y0-m0*x0) - lhc0 + x0**2 + y0**2
                        x_bests = np.roots([x_pw2_coeff, x_pw1_coeff, x_pw0_coeff]) #possible xhcs

                        if printDebugInfoToScreen: print(f"x_bests = {x_bests}")

                        #xhc1, xhc2 = real_xBests = [np.real(x) for x in x_bests if np.isreal(x)]  # returns real part if number is real - discards the 0j part
                        #WARNING (Possiblity of Error - logical error at user level (i.e. the math might be wrong if we use this approach))
                        # Extracting just the real part (not ideal, but it should be the correct approach in this case as we should not get imaginary values unless there are errors due to numerical methods.)
                        xhc1, xhc2 = real_xBests = [np.real(x) for x in x_bests]  # returns real part if number is real - discards the 0j part - sometimes it is of the order of je-5, which is very small but np considers it to be imaginary.

                        #if printDebugInfoToScreen: print(f"\nCoeffs are x^2: {x_pw2_coeff}, x^1: {x_pw1_coeff}, x^0: {x_pw0_coeff}")
                        #if printDebugInfoToScreen: print(f"\nRoots: Xhcs are {xhc1} and {xhc2}")

                        #the possible ys for the two possible xs
                        yhc1 = (y0 - m0*x0) + m0*xhc1
                        yhc2 = (y0 - m0*x0) + m0*xhc2

                        #chose the point closest to tail head or other near points OR farthest from withers or other far points
                        nearPoint = visible_KPs['tail_head'] if 'tail_head' in visible_KPs else visible_KPs['left_pin_bone'] if 'left_pin_bone' in visible_KPs else visible_KPs['right_pin_bone'] if 'right_pin_bone' in visible_KPs else None
                        farPoint  = visible_KPs['withers'] if 'withers' in visible_KPs else visible_KPs['left_shoulder'] if 'left_shoulder' in visible_KPs else visible_KPs['right_shoulder'] if 'right_shoulder' in visible_KPs else visible_KPs['center_back'] if 'center_back' in visible_KPs else None

                        hipConnector = None
                        if nearPoint is not None:
                            if printDebugInfoToScreen: print(f"Using nearpoint {nearPoint}")
                            xnp, ynp = nearPoint
                            len1 = np.sqrt((yhc1-ynp)**2 + (xhc1-xnp)**2)
                            len2 = np.sqrt((yhc2-ynp)**2 + (xhc2-xnp)**2)

                            #pick the point yielding smaller length
                            hipConnector = (xhc1, yhc1) if len1 < len2 else (xhc2, yhc2) #correct

                        elif farPoint is not None:
                            if printDebugInfoToScreen: print(f"Using farpoint {farPoint}")
                            xfp, yfp = farPoint
                            len1 = np.sqrt((yhc1-yfp)**2 + (xhc1-xfp)**2)
                            len2 = np.sqrt((yhc2-yfp)**2 + (xhc2-xfp)**2)

                            #pick the point yielding larger length
                            hipConnector = (xhc1, yhc1) if len1 > len2 else (xhc2, yhc2)

                        else: #when only the two hip bones are visible, this is the best possible estimate
                            hipConnector = (x0, y0)

                        if printDebugInfoToScreen: print(f"predicted Hip connector location = {hipConnector}")
                        #pdb.set_trace()
                        #if printDebugInfoToScreen: print(f"Distance between hip connector and P0 = {np.sqrt((y0-hipConnector[1])**2 + (x0-hipConnector[0])**2)}")
                        if hipConnector is not None:
                            visible_KPs['hip_connector'] = hipConnector
                            instance_KPs[keypoint_names.index('hip_connector')] = [hipConnector[0], hipConnector[1], keypoint_threshold+0.1]
                            break

                    #if a hip bone is not visible, req kps - one hip bone and other spine points, both pin bones
                    elif ("left_hip_bone" in visible_KPs or "right_hip_bone" in visible_KPs) and ('tail_head' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs) and ("left_pin_bone" in visible_KPs and "right_pin_bone" in visible_KPs):
                        # logic:
                        # Fit a second order curve through the three visible spine points.
                        # compute perpendicular distance from visible hip bone to line joining two pin bones  = d1
                        # predicted hip connector is the point on the second degree curve at a distance of d1 from tail head (you could work with the mid point of the two pin bones also)

                        p1 = x1, y1 = visible_KPs['tail_head']
                        p2 = x2, y2 = visible_KPs['center_back']
                        p3 = x3, y3 = visible_KPs['withers']
                        p4 = x4, y4 = visible_KPs['left_hip_bone'] if "left_hip_bone" in visible_KPs else visible_KPs['right_hip_bone']
                        p5 = x5, y5 = visible_KPs['left_pin_bone']
                        p6 = x6, y6 = visible_KPs['right_pin_bone']

                        n = np.array([1, 1, 1])

                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3], [y1, y2, y3], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                        x_pw2_coeff_ulim = 12e-5  # 1e-4 #upper limit of x^2's coefficient # not using any limits for now

                        #we need the spine line to be straight so that the reflected point doesn't go out of the cow or comes inside the cow
                        if a <= x_pw2_coeff_ulim:
                            if printDebugInfoToScreen: print(f"Fixing hip connector: Method 2 - using 2nd degree curve")

                            d1 = get_perpendicularDist_pt_to_line(line_pt1=p5, line_pt2=p6, pt=p4)
                            inter_hipCon = predict_point_on_second_degree_curve(curve=curve, srcPt=p1, distFromSrc=d1, nearPoint=p2) #the predicted point should be towards center_back (p2)

                            if inter_hipCon is not None:
                                visible_KPs['hip_connector'] = inter_hipCon
                                instance_KPs[keypoint_names.index('hip_connector')] = [inter_hipCon[0], inter_hipCon[1], keypoint_threshold + 0.1]
                                logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated hip_connector: {inter_hipCon}.")
                                break

                            else:
                                if printDebugInfoToScreen: print(f"Could not interpolate hip_connector") # as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                                logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate hip_connector") # as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                        else:
                            if printDebugInfoToScreen: print(f"Could not interpolate hip_connector as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate hip_connector as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")


                #missin tail head; required KPs: both pin bones
                elif kpName == "tail_head":
                    if printDebugInfoToScreen: print(f"Fixing tail head")
                    logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing tail head.")

                    #leftpin--------------X---------------right pin
                    #logic - mid point of line joining two pin bones is the interpolated tail head

                    if "left_pin_bone" in visible_KPs and "right_pin_bone" in visible_KPs:

                        x1, y1 = visible_KPs['left_pin_bone']
                        x2, y2 = visible_KPs['right_pin_bone']
                        x0 = (x1+x2)/2; y0 = (y1+y2)/2
                        inter_tailHead = (x0,y0) #interpolated tail head

                        visible_KPs['tail_head'] = inter_tailHead
                        instance_KPs[keypoint_names.index('tail_head')] = [inter_tailHead[0], inter_tailHead[1], keypoint_threshold + 0.1]
                        break

                    #if a pin bone is not visible
                    elif("left_pin_bone" in visible_KPs or "right_pin_bone" in visible_KPs) and ('hip_connector' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs) and ("left_hip_bone" in visible_KPs and "right_hip_bone" in visible_KPs):
                        # logic:
                        # Fit a second order curve through the three visible spine points.
                        # compute perpendicular distance from visible pin bone to line joining two hip bones  = d1
                        # predicted tail head is the point on the second degree curve at a distance of d1 from hip connector (you could work with the mid point of the two pin bones also)

                        p1 = x1, y1 = visible_KPs['hip_connector']
                        p2 = x2, y2 = visible_KPs['center_back']
                        p3 = x3, y3 = visible_KPs['withers']
                        p4 = x4, y4 = visible_KPs['left_pin_bone'] if "left_pin_bone" in visible_KPs else visible_KPs['right_pin_bone']
                        p5 = x5, y5 = visible_KPs['left_hip_bone']
                        p6 = x6, y6 = visible_KPs['right_hip_bone']

                        n = np.array([1, 1, 1])

                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3], [y1, y2, y3], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        x_pw2_coeff_ulim = 12e-5  # 1e-4 #upper limit of x^2's coefficient # not using any limits for now

                        #we need the spine line to be straight so that the reflected point doesn't go out of the cow or comes inside the cow
                        if a <= x_pw2_coeff_ulim:
                            if printDebugInfoToScreen: print(f"Fixing tail head: Method 2 - using 2nd degree curve")

                            d1 = get_perpendicularDist_pt_to_line(line_pt1=p5, line_pt2=p6, pt=p4)
                            inter_tailHead = predict_point_on_second_degree_curve(curve=curve, srcPt=p1, distFromSrc=d1, farPoint=p2) #the predicted point should be further from center_back (p2)

                            if inter_tailHead is not None:
                                visible_KPs['tail_head'] = inter_tailHead
                                instance_KPs[keypoint_names.index('tail_head')] = [inter_tailHead[0], inter_tailHead[1], keypoint_threshold + 0.1]
                                logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated tail_head: {inter_tailHead}.")
                                break

                            else:
                                if printDebugInfoToScreen: print(f"Could not interpolate tail_head") # as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                                logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate tail_head") # as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                        else:
                            if printDebugInfoToScreen: print(f"Could not interpolate tail_head as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate tail_head as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")

                #missing left pin bones;
                elif kpName == "left_pin_bone":
                    # required KPs: right pin bone, tail head and hip connector
                    if  'hip_connector' in visible_KPs and 'tail_head' in visible_KPs and 'right_pin_bone' in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing left pin bone")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left pin bone.")

                        #logic:
                        #using the mid point formula from tail head interpolation doesn't work
                        #should reflect the other pin bone across line through hip connector and tail head

                        #l12 - line through hip con and tail head
                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']
                        p3 = visible_KPs['right_pin_bone']
                        x4, y4 = reflect_point_across_line(p1=p1, p3=p3, p2=p2)

                        inter_left_pin_bone = (x4, y4) #interpolated left pin bone
                        #if printDebugInfoToScreen: print(f"interpolated left pin bone = {inter_left_pin_bone}")

                        visible_KPs['left_pin_bone'] = inter_left_pin_bone
                        instance_KPs[keypoint_names.index('left_pin_bone')] = [inter_left_pin_bone[0], inter_left_pin_bone[1], keypoint_threshold + 0.1]

                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated left_pin_bone: {inter_left_pin_bone}.")
                        break


                    # Alternate method: required KPs: either hip bone, tail head and hip connector. Useful when both pin bones are not detected.
                    elif ('tail_head' in visible_KPs and 'hip_connector' in visible_KPs) and ('left_hip_bone' in visible_KPs or 'right_hip_bone' in visible_KPs):
                        if printDebugInfoToScreen: print(f"Fixing left pin bone with Method 2.")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left pin bone.")

                        # logic:
                        # the required pin bone is the point on the perpendicular to line joining hip connecetor and tail head dropped at the tail head at a given distance, closest to near point or farthest from far point
                        # this distance = 0.45 * perpendicular distance from given hip bone to line joining hip connector and tail head

                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']

                        if printDebugInfoToScreen: print(f"p1 = {p1}, p2 = {p2}")

                        # check if both p1 and p2 are detected in the same place
                        # This error actually occurs. When this happens, we cannot determine a line through just one point.
                        if p1 == p2:
                            if printDebugInfoToScreen: print(f"Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            break

                        nearPoint = None; farPoint = None; p3 = []

                        if 'left_hip_bone' in visible_KPs:
                            nearPoint = p3 = visible_KPs['left_hip_bone']
                        elif 'right_hip_bone' in visible_KPs:
                            farPoint = p3 = visible_KPs['right_hip_bone']

                        l1 = get_perpendicularDist_pt_to_line(line_pt1=p1, line_pt2=p2, pt=p3)

                        ratio = 0.35 #0.4 #0.45 # hyper parameter - from heuristics
                        l2 = ratio*l1

                        #if printDebugInfoToScreen: print(f"l1 = {l1}, samller length l2 = {l2}")

                        #if printDebugInfoToScreen: print(f"LPin method2: Near point = {nearPoint}, far point = {farPoint}")

                        x4, y4 = get_point_at_given_dist_on_perp_to_line(pt1=p1, pt2=p2, dist=l2, nearPoint=nearPoint, farPoint=farPoint) # either nearPoint or farPoint will be None

                        inter_left_pin_bone = (x4, y4)  # interpolated left pin bone
                        # if printDebugInfoToScreen: print(f"interpolated left pin bone = {inter_left_pin_bone}")

                        visible_KPs['left_pin_bone'] = inter_left_pin_bone
                        instance_KPs[keypoint_names.index('left_pin_bone')] = [inter_left_pin_bone[0], inter_left_pin_bone[1], keypoint_threshold + 0.1]

                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated left_pin_bone: {inter_left_pin_bone}.")

                        break

                #missing right pin bone; required KPs: left pin bone, tail head and hip connector
                elif kpName == "right_pin_bone":
                    if "hip_connector" in visible_KPs and 'tail_head' in visible_KPs and 'left_pin_bone' in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing right pin bone")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right pin bone.")

                        # logic:
                        # using the mid point formula from tail head interpolation doesn't work
                        # should reflect the other pin bone across line through hip connector and tail head

                        # l12 - line through hip con and tail head
                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']
                        p3 = visible_KPs['left_pin_bone']
                        x4, y4 = reflect_point_across_line(p1=p1, p3=p3, p2=p2)

                        inter_right_pin_bone = (x4, y4)  # interpolated left pin bone
                        #if printDebugInfoToScreen: print(f"interpolated right pin bone = {inter_right_pin_bone}")

                        visible_KPs['right_pin_bone'] = inter_right_pin_bone
                        instance_KPs[keypoint_names.index('right_pin_bone')] = [inter_right_pin_bone[0], inter_right_pin_bone[1], keypoint_threshold + 0.1]

                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated right_pin_bone: {inter_right_pin_bone}.")
                        break

                    # Alternate method: required KPs: either hip bone, tail head and hip connector. Useful when both pin bones are not detected.
                    # This alternate method is redundant if used for both pin bones. If we interpolate the first pin bone using the alternate method, the second pin bone could be interpolated in the next iteration using the first method.
                    # But I put it here anyway for the sake of uniformity.
                    elif ('tail_head' in visible_KPs and 'hip_connector' in visible_KPs) and ('left_hip_bone' in visible_KPs or 'right_hip_bone' in visible_KPs):
                        if printDebugInfoToScreen: print(f"Fixing right pin bone with Method 2.")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right pin bone.")

                        # logic:
                        # the required pin bone is the point on the perpendicular to line joining hip connecetor and tail head dropped at the tail head at a given distance, closest to near point or farthest from far point
                        # this distance = 0.45 * perpendicular distance from given hip bone to line joining hip connector and tail head

                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']

                        # check if both p1 and p2 are detected in the same place
                        # This error actually occurs. When this happens, we cannot determine a line through just one point.
                        if p1 == p2:
                            if printDebugInfoToScreen: print(f"Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            break

                        nearPoint = None;
                        farPoint = None;
                        p3 = []

                        if 'right_hip_bone' in visible_KPs:
                            nearPoint = p3 = visible_KPs['right_hip_bone']
                        elif 'left_hip_bone' in visible_KPs:
                            farPoint = p3 = visible_KPs['left_hip_bone']

                        l1 = get_perpendicularDist_pt_to_line(line_pt1=p1, line_pt2=p2, pt=p3)

                        ratio = 0.35 #0.4 #0.45  # hyper parameter - from heuristics
                        l2 = ratio * l1

                        #if printDebugInfoToScreen: print(f"l1 = {l1}, samller length l2 = {l2}")

                        #if printDebugInfoToScreen: print(f"RPin method2: Near point = {nearPoint}, far point = {farPoint}")

                        x4, y4 = get_point_at_given_dist_on_perp_to_line(pt1=p1, pt2=p2, dist=l2, nearPoint=nearPoint,
                                                                         farPoint=farPoint)  # either nearPoint or farPoint will be None

                        inter_right_pin_bone = (x4, y4)  # interpolated right pin bone
                        # if printDebugInfoToScreen: print(f"interpolated right pin bone = {inter_right_pin_bone}")

                        visible_KPs['right_pin_bone'] = inter_right_pin_bone
                        instance_KPs[keypoint_names.index('right_pin_bone')] = [inter_right_pin_bone[0],
                                                                               inter_right_pin_bone[1],
                                                                               keypoint_threshold + 0.1]

                        logging.debug(
                            f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated right_pin_bone: {inter_right_pin_bone}.")

                        break


                #missing left hip bone;
                elif kpName == "left_hip_bone":
                    # required KPs: hip connector, tail head, right hip bone
                    if  'hip_connector' in visible_KPs and 'tail_head' in visible_KPs and 'right_hip_bone' in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing left hip bone")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left hip bone.")

                        #logic:
                        #should reflect the other hip bone across line through hip connector and tail head

                        #l12 - line through hip con and tail head
                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']
                        p3 = visible_KPs['right_hip_bone']
                        x4, y4 = reflect_point_across_line(p1=p1, p3=p3, p2=p2)

                        inter_left_hip_bone = (x4, y4) #interpolated left pin bone
                        #if printDebugInfoToScreen: print(f"interpolated left hip bone = {inter_left_hip_bone}")

                        visible_KPs['left_hip_bone'] = inter_left_hip_bone
                        instance_KPs[keypoint_names.index('left_hip_bone')] = [inter_left_hip_bone[0], inter_left_hip_bone[1], keypoint_threshold + 0.1]

                        logging.debug(
                            f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated left_hip_bone: {inter_left_hip_bone}.")
                        break


                    # Alternate method: required KPs: either pin bone, tail head and hip connector
                    # Useful when both hip bones are not detected!
                    elif ('tail_head' in visible_KPs and 'hip_connector' in visible_KPs) and ('left_pin_bone' in visible_KPs or 'right_pin_bone' in visible_KPs):
                        if printDebugInfoToScreen: print(f"Fixing left hip bone with Method 2.")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left pin bone.")

                        # logic:
                        # the required pin bone is the point on the perpendicular to line joining hip connecetor and tail head dropped at the tail head at a given distance, closest to near point or farthest from far point
                        # this distance = (1/0.45) * perpendicular distance from given pin bone to line joining hip connector and tail head

                        p1 = visible_KPs['tail_head']
                        p2 = visible_KPs['hip_connector']

                        # check if both p1 and p2 are detected in the same place
                        # This error actually occurs. When this happens, we cannot determine a line through just one point.
                        if p1 == p2:
                            if printDebugInfoToScreen: print(f"Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            break

                        nearPoint = None;
                        farPoint = None;
                        p3 = []

                        if 'left_pin_bone' in visible_KPs:
                            nearPoint = p3 = visible_KPs['left_pin_bone']
                        elif 'right_pin_bone' in visible_KPs:
                            farPoint = p3 = visible_KPs['right_pin_bone']

                        if printDebugInfoToScreen: print(f"p1 = {p1}, p2 = {p2}, p3 = {p3}")

                        l1 = get_perpendicularDist_pt_to_line(line_pt1=p1, line_pt2=p2, pt=p3)

                        ratio = 1/0.35 #0.4  # hyper parameter - from heuristics
                        l2 = ratio * l1


                        #if printDebugInfoToScreen: print(f"l1 = {l1}, longer length l2 = {l2}")
                        #if printDebugInfoToScreen: print(f"LHip method2: Near point = {nearPoint}, far point = {farPoint}")

                        x4, y4 = get_point_at_given_dist_on_perp_to_line(pt1=p1, pt2=p2, dist=l2, nearPoint=nearPoint,farPoint=farPoint)  # either nearPoint or farPoint will be None

                        inter_left_hip_bone = (x4, y4)  # interpolated left hip bone
                        # if printDebugInfoToScreen: print(f"interpolated left hip bone = {inter_left_hip_bone}")

                        visible_KPs['left_hip_bone'] = inter_left_hip_bone
                        instance_KPs[keypoint_names.index('left_hip_bone')] = [inter_left_hip_bone[0],
                                                                               inter_left_hip_bone[1],
                                                                               keypoint_threshold + 0.1]

                        logging.debug(
                            f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated left_hip_bone: {inter_left_hip_bone}.")

                        break

                    # when hip connector is not visible
                    '''
                    elif ('tail_head' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs) and 'right_hip_bone' in visible_KPs:

                        # logic:
                        # Fit a second order curve through the three visible spine points.
                        # Only if it is a straight line or is very close to a straight line, reflect the known hip bone across the line joining center back and tail head
                        # otherwise, do not attempt interpolation - ratio techniques might not work as center back is detected with high variance

                        p1 = x1, y1 = visible_KPs['tail_head']
                        p2 = x2, y2 = visible_KPs['center_back']
                        p3 = x3, y3 = visible_KPs['withers']
                        p4 = x4, y4 = visible_KPs['right_hip_bone']

                        n = np.array([1, 1, 1])
                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3], [y1, y2, y3], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break
    
                        x_pw2_coeff_ulim = 12e-5  # 1e-4 #upper limit of x^2's coefficient

                        if a < x_pw2_coeff_ulim:  # i.e. if curve is more like a straight line
                            if printDebugInfoToScreen: print(f"Fixing left hip bone: Method 3")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left hip bone (Method 3).")

                            inter_left_hip_bone = reflect_point_across_line(p1=p1, p3=p4, p2=p2)
                            visible_KPs['left_hip_bone'] = inter_left_hip_bone
                            instance_KPs[keypoint_names.index('left_hip_bone')] = [inter_left_hip_bone[0],
                                                                                   inter_left_hip_bone[1],
                                                                                   keypoint_threshold + 0.1]

                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated inter_left_hip_bone: {inter_left_hip_bone}.")
                            break
                        else:
                            if printDebugInfoToScreen: print(
                                f"Could not interpolate left_hip_bone as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(
                                f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate left_hip_bone as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                    '''

                #missing right hip bone;
                elif kpName == "right_hip_bone":
                    # required KPs: hip connector, tail head, left hip bone
                    if 'hip_connector' in visible_KPs and 'tail_head' in visible_KPs and 'left_hip_bone' in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing right hip bone")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right hip bone.")

                        # logic:
                        # should reflect the other hip bone across line through hip connector and tail head

                        # l12 - line through hip con and tail head
                        p1 = visible_KPs['hip_connector']
                        p2 = visible_KPs['tail_head']
                        p3 = visible_KPs['left_hip_bone']
                        x4, y4 = reflect_point_across_line(p1=p1, p3=p3, p2=p2) #p4

                        inter_right_hip_bone = (x4, y4)  # interpolated right hip bone
                        #if printDebugInfoToScreen: print(f"interpolated right hip bone = {inter_right_hip_bone}")

                        visible_KPs['right_hip_bone'] = inter_right_hip_bone
                        instance_KPs[keypoint_names.index('right_hip_bone')] = [inter_right_hip_bone[0], inter_right_hip_bone[1], keypoint_threshold + 0.1]

                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated right_hip_bone: {inter_right_hip_bone}.")
                        break

                    # Alternate method: required KPs: either pin bone, tail head and hip connector
                    # Useful when both hip bones are not detected!
                    # This alternate method is redundant if used for both hip bones. If we interpolate the first hip bone using the alternate method, the second hip bone could be interpolated in the next iteration using the first method.
                    # But I put it here anyway for the sake of uniformity.
                    elif ('tail_head' in visible_KPs and 'hip_connector' in visible_KPs) and ('left_pin_bone' in visible_KPs or 'right_pin_bone' in visible_KPs):
                        if printDebugInfoToScreen: print(f"Fixing right hip bone with Method 2.")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right pin bone.")

                        # logic:
                        # the required pin bone is the point on the perpendicular to line joining hip connecetor and tail head dropped at the tail head at a given distance, closest to near point or farthest from far point
                        # this distance = (1/0.45) * perpendicular distance from given pin bone to line joining hip connector and tail head

                        p1 = visible_KPs['tail_head']
                        p2 = visible_KPs['hip_connector']

                        # check if both p1 and p2 are detected in the same place
                        # This error actually occurs. When this happens, we cannot determine a line through just one point.
                        if p1 == p2:
                            if printDebugInfoToScreen: print(f"Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolation aborted. Tail head and hip connector detected at the same point ({p1}).")
                            break

                        nearPoint = None; farPoint = None; p3 = []

                        if 'right_pin_bone' in visible_KPs:
                            nearPoint = p3 = visible_KPs['right_pin_bone']
                        elif 'left_pin_bone' in visible_KPs:
                            farPoint = p3 = visible_KPs['left_pin_bone']

                        l1 = get_perpendicularDist_pt_to_line(line_pt1=p1, line_pt2=p2, pt=p3)

                        ratio = 1 / 0.35 #0.4  # hyper parameter - from heuristics
                        l2 = ratio * l1

                        #if printDebugInfoToScreen: print(f"p1 = {p1}, p2 = {p2}, p3 = {p3}")
                        #if printDebugInfoToScreen: print(f"l1 = {l1}, longer length l2 = {l2}")
                        #if printDebugInfoToScreen: print(f"RHip method2: Near point = {nearPoint}, far point = {farPoint}")

                        x4, y4 = get_point_at_given_dist_on_perp_to_line(pt1=p1, pt2=p2, dist=l2, nearPoint=nearPoint,
                                                                         farPoint=farPoint)  # either nearPoint or farPoint will be None

                        inter_right_hip_bone = (x4, y4)  # interpolated right hip bone
                        # if printDebugInfoToScreen: print(f"interpolated right hip bone = {inter_right_hip_bone}")

                        visible_KPs['right_hip_bone'] = inter_right_hip_bone
                        instance_KPs[keypoint_names.index('right_hip_bone')] = [inter_right_hip_bone[0],
                                                                               inter_right_hip_bone[1],
                                                                               keypoint_threshold + 0.1]

                        logging.debug(
                            f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated right_hip_bone: {inter_right_hip_bone}.")

                        break

                    # when hip connector is not visible
                    '''
                    elif ('tail_head' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs) and 'left_hip_bone' in visible_KPs:

                        # logic:
                        # Fit a second order curve through the three visible spine points.
                        # Only if it is a straight line or is very close to a straight line, reflect the known hip bone across the line joining center back and tail head
                        # otherwise, do not attempt interpolation - ratio techniques might not work as center back is detected with high variance

                        p1 = x1, y1 = visible_KPs['tail_head']
                        p2 = x2, y2 = visible_KPs['center_back']
                        p3 = x3, y3 = visible_KPs['withers']
                        p4 = x4, y4 = visible_KPs['left_hip_bone']

                        n = np.array([1, 1, 1])
                        
                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3], [y1, y2, y3], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                            if printDebugInfoToScreen: print(f"a,b,c = {a}, {b}, {c}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break
                            
                            
                        x_pw2_coeff_ulim = 12e-5  # 1e-4 #upper limit of x^2's coefficient

                        if a < x_pw2_coeff_ulim:  # i.e. if curve is more like a straight line
                            if printDebugInfoToScreen: print(f"Fixing right hip bone: Method 3")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right hip bone (Method 3).")

                            inter_right_hip_bone = reflect_point_across_line(p1=p1, p3=p4, p2=p2)
                            visible_KPs['right_hip_bone'] = inter_right_hip_bone
                            instance_KPs[keypoint_names.index('right_hip_bone')] = [inter_right_hip_bone[0],
                                                                                   inter_right_hip_bone[1],
                                                                                   keypoint_threshold + 0.1]

                            logging.debug(
                                f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated inter_right_hip_bone: {inter_right_hip_bone}.")
                            break
                        else:
                            if printDebugInfoToScreen: print(
                                f"Could not interpolate right_hip_bone as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(
                                f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate right_hip_bone as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                    '''


                #missing withers; required KPs: tail head, hip connector, center back, left and right shoulders
                elif kpName == "withers":
                    if "center_back" in visible_KPs and "hip_connector" in visible_KPs and "tail_head" in visible_KPs and "left_shoulder" in visible_KPs and "right_shoulder" in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing withers")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing withers.")

                        #logic: fit 2nd degree polynomial through the spine points and the mid point of line joining left and right shoulders

                        x1, y1 = visible_KPs['tail_head']
                        x2, y2 = visible_KPs['hip_connector']
                        x3, y3 = visible_KPs['center_back']
                        x41, y41 = visible_KPs['left_shoulder']
                        x42, y42 = visible_KPs['right_shoulder']

                        #point p4 = (x4,y4)
                        x4 = (x41+x42)/2; y4 = (y41+y42)/2

                        n = np.array([1, 2, 3, 1])

                        try:
                            curve =  np.polyfit([x1,x2,x3,x4], [y1,y2,y3,y4],  w=np.sqrt(n), deg=2) #ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Obtained coeffs = {curve}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        #select a point on this curve such that it has a distance from the center back equal to percentDeviation times the distancce between center back and the mid point of line joining the two shoulders points, away from the center back
                        #len_cback_p4 = np.sqrt((y4-y3)**2 + (x4-x3)**2)
                        #percentDeviation = 0.9
                        #len_p4_withers = percentDeviation * len_cback_p4

                        #new reference point
                        len_hipCon_cback = np.sqrt((y3-y2)**2 + (x3-x2)**2)
                        percentDeviation = 2/3 #0.5
                        len_cback_withers = percentDeviation * len_hipCon_cback

                        farPoint = visible_KPs['hip_connector'] #visible_KPs['center_back']

                        #withersPt = predict_point_on_second_degree_curve(curve=curve, srcPt=(x4,y4), distFromSrc=len_p4_withers, farPoint=farPoint)
                        withersPt = predict_point_on_second_degree_curve(curve=curve, srcPt=(x3,y3), distFromSrc=len_cback_withers, farPoint=farPoint)

                        if withersPt is not None:
                            visible_KPs['withers'] = withersPt
                            instance_KPs[keypoint_names.index('withers')] = [withersPt[0], withersPt[1], keypoint_threshold + 0.1]

                        #return instance_KPs, curve #remove later
                        break

                #missing center back; required KPs: tail head, hip connector, withers, left and right shoulders
                elif kpName == "center_back":
                    if "tail_head" in visible_KPs and "hip_connector" in visible_KPs and "withers" in visible_KPs and "left_shoulder" in visible_KPs and "right_shoulder" in visible_KPs:
                        if printDebugInfoToScreen: print(f"Fixing center back")
                        logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing center back.")

                        x1, y1 = visible_KPs['tail_head']
                        x2, y2 = visible_KPs['hip_connector']
                        x4, y4 = visible_KPs['withers']
                        x31, y31 = visible_KPs['left_shoulder']
                        x32, y32 = visible_KPs['right_shoulder']

                        len_withers_leftShldr = np.sqrt((y31-y4)**2 +  (x31-x4)**2)
                        len_withers_rightShldr = np.sqrt((y32-y4)**2 +  (x32-x4)**2)

                        # point p3 = (x3,y3)
                        #x3 = (x31 + x32) / 2; y3 = (y31 + y32) / 2 #simple average - not good enough
                        #going for weighted average - this is equal to simple average if the lengths are equal
                        #the curve should be pulled towards the side that is less visible - towards the shoulder point closer to the withers!
                        avg_exp= 5.5 #5 #4 #1 #exponenet used in averaging
                        x3 = (len_withers_rightShldr**avg_exp * x31 + len_withers_leftShldr**avg_exp * x32) / (len_withers_leftShldr**avg_exp +  len_withers_rightShldr**avg_exp + 1e-10) # adding 1e-10 to denominator to avoid div by zero error
                        y3 = (len_withers_rightShldr**avg_exp * y31 + len_withers_leftShldr**avg_exp * y32) / (len_withers_leftShldr**avg_exp +  len_withers_rightShldr**avg_exp + 1e-10) # adding 1e-10 to denominator to avoid div by zero error

                        #n = np.array([1, 1, 1, 1]) # works better #n = np.array([1, 1, 2, 1]) # might not work well
                        n = np.array([1, 2, 3, 2]) # works better #n = np.array([1, 1, 2, 1]) # also works well

                        try:
                            #curve = np.polyfit([x1, x2, x32, x4], [y1, y2, y32, y4], w=np.sqrt(n), deg=2)  # ax^2 + bx + c #wrong but curve is right for test case
                            curve = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], w=np.sqrt(n), deg=2)  # ax^2 + bx + c #actual correct set of points
                            if printDebugInfoToScreen: print(f"Obtained coeffs = {curve}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        len_hipCon_withers = np.sqrt((y2-y4)**2 + (x2-x4)**2)
                        fractionalDeviation = 2/3
                        len_hipCon_cback = fractionalDeviation * len_hipCon_withers

                        #predicted cback should be at a distance of 2/3rd the distance between hip connector and withers from the hip connector; and, furthest from the tail head
                        farPoint = visible_KPs['tail_head']
                        cbackPt = predict_point_on_second_degree_curve(curve=curve, srcPt=(x2, y2), distFromSrc=len_hipCon_cback, farPoint=farPoint)

                        if cbackPt is not None:
                            visible_KPs['center_back'] = cbackPt
                            instance_KPs[keypoint_names.index('center_back')] = [cbackPt[0], cbackPt[1], keypoint_threshold + 0.1]

                        #return instance_KPs, curve
                        break

                # not that reliable - works only if spine is straight, so, this condition should be checked at the end - after all other kps (like how it is being done here)
                #missing left shoulder: required KPs: tail head, hip connector, center back, withers, the other shoulder
                elif kpName == "left_shoulder":
                    if 'tail_head' in visible_KPs and 'hip_connector' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs and 'right_shoulder' in visible_KPs:

                        # logic:
                        # Fit a second order curve through the four spine points.
                        # Only if it is a straight line or is very close to a straight line, reflect the known shoulder across the line joining center back and withers
                        # Otherwise, do not try to interpolate the shoulders - this is a highly deformable region
                        # and techniques used to interpolate shoulders may not detect the shoulders in the right place most times.

                        p1 = x1, y1 = visible_KPs['tail_head']
                        p2 = x2, y2 = visible_KPs['hip_connector']
                        p3 = x3, y3 = visible_KPs['center_back']
                        p4 = x4, y4 = visible_KPs['withers']
                        p5 = x5, y5 = visible_KPs['right_shoulder']

                        n = np.array([1, 1, 1, 1])

                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        x_pw2_coeff_ulim = 12e-5 #1e-4 #upper limit of x^2's coefficient
                        if a < x_pw2_coeff_ulim: # i.e. if curve is more like a straight line
                            if printDebugInfoToScreen: print(f"Fixing left shoulder")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing left shoulder.")

                            inter_left_shoulder = reflect_point_across_line(p1=p3, p3=p5, p2=p4)
                            visible_KPs['left_shoulder'] = inter_left_shoulder
                            instance_KPs[keypoint_names.index('left_shoulder')] = [inter_left_shoulder[0], inter_left_shoulder[1], keypoint_threshold + 0.1]

                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated left_shoulder: {inter_left_shoulder}.")
                            break
                        else:
                            if printDebugInfoToScreen: print(f"Could not interpolate left_shoulder as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate left_shoulder as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")

                # not that reliable - works only if spine is straight, so, this condition should be checked at the end - after all other kps (like how it is being done here)
                # missing right shoulder: required KPs: tail head, hip connector, center back, withers, the other shoulder
                elif kpName == "right_shoulder":
                    if 'tail_head' in visible_KPs and 'hip_connector' in visible_KPs and 'center_back' in visible_KPs and 'withers' in visible_KPs and 'left_shoulder' in visible_KPs:

                        # logic:
                        # Fit a second order curve through the four spine points.
                        # Only if it is a straight line or is very close to a straight line, reflect the known shoulder across the line joining center back and withers
                        # Otherwise, do not try to interpolate the shoulders - this is a highly deformable region
                        # and techniques used to interpolate shoulders may not detect the shoulders in the right place most times.

                        p1 = x1, y1 = visible_KPs['tail_head']
                        p2 = x2, y2 = visible_KPs['hip_connector']
                        p3 = x3, y3 = visible_KPs['center_back']
                        p4 = x4, y4 = visible_KPs['withers']
                        p5 = x5, y5 = visible_KPs['left_shoulder']

                        n = np.array([1, 1, 1, 1])

                        try:
                            a, b, c = curve = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], w=np.sqrt(n), deg=2)  # ax^2 + bx + c
                            if printDebugInfoToScreen: print(f"Coeff of x^2 = {a}")
                        except Exception as e:
                            if printDebugInfoToScreen: print(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            logging.debug(f"Obtained {e} - Could not fit curve through points. Breaking.")
                            break

                        x_pw2_coeff_ulim = 12e-5 #1e-4  # upper limit of x^2's coefficient
                        if a < x_pw2_coeff_ulim:  # i.e. if curve is more like a straight line
                            if printDebugInfoToScreen: print(f"Fixing right shoulder")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Fixing right shoulder.")

                            inter_right_shoulder = reflect_point_across_line(p1=p3, p3=p5, p2=p4)
                            visible_KPs['right_shoulder'] = inter_right_shoulder
                            instance_KPs[keypoint_names.index('right_shoulder')] = [inter_right_shoulder[0], inter_right_shoulder[1], keypoint_threshold + 0.1]

                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Interpolated right_shoulder: {inter_right_shoulder}.")
                            break
                        else:
                            if printDebugInfoToScreen: print(f"Could not interpolate right_shoulder as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")
                            logging.debug(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Could not interpolate right_shoulder as spine is not straight: x^2 coeff = {a} < ulim {x_pw2_coeff_ulim}")


        if printDebugInfoToScreen: print(f"Current keypoint values are \n{instance_KPs}")

        n_visibleKPs = len(visible_KPs)
        if n_visibleKPs == prev_n_visibleKPs:
            if printDebugInfoToScreen: print(f"Breaking out of missing KP checker - while loop")
            break
        else:
            prev_n_visibleKPs = n_visibleKPs
            if printDebugInfoToScreen: print(f"\nGoing for another iteration of missing KP checker - while loop: prev_n_visibleKPs = {prev_n_visibleKPs}\n")

    # restrict points to within frame
    # checking if KPs are inside the frame. If not, we make them invisible.
    # We check all of them at the end because, we could use the invisible KPs to interpolate other visible KPs, and they could be useful in the future.
    for idx, vals in enumerate(instance_KPs):
        kp_x, kp_y, visibility = vals
        if not ((0<=kp_x<frameW) and (0<=kp_y<frameH)):
            instance_KPs[idx] = [kp_x, kp_y, keypoint_threshold-0.1]
            if printDebugInfoToScreen: print(f"Keypoint {keypoint_names[idx]} at loc ({kp_x},{kp_y}) suppressed as it is out of frame bounds.")
            logging.info(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Keypoint {keypoint_names[idx]} at loc ({kp_x},{kp_y}) suppressed as it is out of frame bounds.")
        else:
            # if printDebugInfoToScreen: print(f"Keypoint {keypoint_names[idx]} at loc ({kp_x},{kp_y}) is within frame bounds.") # Avoiding too many prints.
            pass

    return  instance_KPs

def get_perpendicularDist_pt_to_line(line_pt1,line_pt2, pt, printDebugInfoToScreen=False):
    '''
    Computes and returns the perpendicular distance between a point pt and a line (through the points line_pt1 and line_pt2)
    :param line_pt1: a point on the line (x_1,y_1)
    :param line_pt2: another point on the line (x_2,y_2)
    :param pt: point not on the line (x,y)
    :return: perpendicular distance from point to line
    '''

    '''
    Distance D = abs(ax+by+c)/sqrt(a^2+b^2)
    pt = (x,y); line => aX+bY+c = 0
    https://www.intmath.com/plane-analytic-geometry/perpendicular-distance-point-line.php
    '''
    x1, y1 = line_pt1; x2, y2 = line_pt2; x, y = pt

    m = (y2 - y1)/((x2 - x1)+ 1e-10) # adding small number to avoid division by 0

    # y - y1 = m(x - x1)
    #y - mx + (m x1 - y1) = 0
    #-mx + y + (m x1 - y1) = 0 <=> Ax + By + C = 0
    a = -1*m; b = 1; c = (m*x1 - y1)

    D = abs(a*x+b*y+c)/np.sqrt(a**2 + b**2)
    D = D.item() #to avoid YAML incompatibility

    return D

def get_point_at_given_dist_on_perp_to_line(pt1, pt2, dist, nearPoint=None, farPoint=None, printDebugInfoToScreen=False):
    '''
    Computes and returns point at a given distance (len) from line (through pt1 and pt2) on perpendicular to the line at pt1,
    that is closest to nearPoint or farthest from farPoint

    :param pt1: point on the line (x1,y1)
    :param pt2: another point on the line (x2, y2)
    :param dist: perpendicular distance of point from the line
    :param nearPoint: reference point (x_np, y_np) - the obtained point must be closer to this point. (Quadratic equation results in 2 solutions).
    :param farPoint: reference point (x_np, y_np) - the obtained point must be farther from this point. (Quadratic equation results in 2 solutions).
    :return:
    '''

    '''
    <---pt1---------------pt2---> line l1 (slope m1)
                           |
                           | (line l2 - slope m2)
      nearPoint            |
                          pt3
    '''
    l = dist #alias
    x1, y1 = pt1; x2, y2 = pt2


    #slope of line l1
    m1 = (y2-y1)/((x2-x1) + 1e-10) # adding small number to avoid division by 0
    m2 = -1/m1 if m1 != 0 else -1/(m1+1e-10) # to avoid div by 0

    #if printDebugInfoToScreen: print(f"(x1,y1) = ({x1},{y1}), (x2,y2) = ({x2},{y2}), m1 = {m1}")

    x_pw2_coeff = (1 + m2**2)
    x_pw1_coeff = -2*x2 * (1+m2**2)
    x_pw0_coeff = x2**2 * (1+m2**2) - l**2



    x_bests = np.roots([x_pw2_coeff, x_pw1_coeff, x_pw0_coeff])
    #if printDebugInfoToScreen: print(f"possible x values = {x_bests}")

    # returns real part - sometimes imaginary part is of the order of 1e-5 due to numerical errors. So we consider only the real parts without checking if it is imaginary.
    # we believe that since all input points are real, the roots should be real (might not be a rigorous reason, but works) (b^2 - 4ac >= 0)
    #on some machines, this numerical error might be more, on some it might be less, so we directly discard the imaginary part
    real_xBests = [np.real(x) for x in x_bests]
    real_yBests = [m2*(x-x2)+y2 for x in real_xBests]

    pt3_x1, pt3_x2 = real_xBests
    pt3_y1, pt3_y2 = real_yBests

    #if printDebugInfoToScreen: print(f"Two possible points at same distance. P1 = {(pt3_x1, pt3_y1)}, P2 = {(pt3_x2, pt3_y2 )}")

    x_pred = []; y_pred = [] #the predicted point p3

    if nearPoint is not None:
        x_np, y_np = nearPoint

        #compare distance to near point
        if ((pt3_x1 - x_np)**2 + (pt3_y1-y_np)**2) < ((pt3_x2 - x_np)**2 + (pt3_y2-y_np)**2):
            x_pred = pt3_x1
            y_pred = pt3_y1
        else:
            x_pred = pt3_x2
            y_pred = pt3_y2

    elif farPoint is not None:
        x_fp, y_fp = farPoint

        # compare distance to far point
        if ((pt3_x1 - x_fp) ** 2 + (pt3_y1 - y_fp) ** 2) > ((pt3_x2 - x_fp) ** 2 + (pt3_y2 - y_fp) ** 2):
            x_pred = pt3_x1
            y_pred = pt3_y1
        else:
            x_pred = pt3_x2
            y_pred = pt3_y2

    else:
        logging.error(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Missing near point and far point. Specify at least one of them.")
        if printDebugInfoToScreen: print(f"ERROR: Missing near point and far point. Specify at least one of them.")

    pt3 = (x_pred, y_pred)

    #if printDebugInfoToScreen: print(f"m2 initial = {m2}, m2 new = {(y_pred-y2)/(x_pred-x2)}")

    return pt3



def predict_point_on_second_degree_curve(curve, srcPt, distFromSrc, farPoint=None, nearPoint=None, printDebugInfoToScreen=False):
    '''
    Finds a point on the given 2nd degree curve such that it is a distance of distFromSrc from the source point srcPt and is nearest to the near point or farthest from the far point.
    Specifying either the near point or far point is enough.

    Use this for predicting withers and center back points.

    :param curve: list or np array of coefficents of 2nd degree polynomial
    :param srcPt: source point (x_src,y_src)
    :param distFromSrc: distance from source pt
    :param farPoint: far point (x_fp, y_fp)
    :param nearPoint: near point (x_np, y_np)
    :return: best point (x_best, y_best)
    '''

    x_src, y_src = srcPt

    # LOGIC:
    # select random point (x,y) on the curve as the best point candidate,
    # set up equation based on given condition that it should be at a distance of distFromSrc from the source point
    # use distance formula - distFromSrc = np.sqrt((y-y_src)**2 + (x-x_src)**2)
    # substitute the value of y from the curve equation --> y = ax**2 + bx + c in the above equation
    # simplify the equation to get it into the form - Ax**4 + Bx**3 + Cx**2 + Dx + E = 0
    # solve it using the np.roots function to get possible values of x
    # reject all imaginary values, with the real values, find the corresponding y coordinates to get the candidates for best point
    # search for the best point out of these candidates based on nearness to nearPoint or farness from farPoint

    a, b, c = curve
    x_pw4_coeff = a ** 2
    x_pw3_coeff = 2 * a * b
    x_pw2_coeff = b ** 2 + 2 * a * c - 2 * a * y_src + 1
    x_pw1_coeff = 2 * b * c - 2 * b * y_src - 2 * x_src
    x_pw0_coeff = c ** 2 + y_src ** 2 - 2 * c * y_src + x_src ** 2 - distFromSrc ** 2

    x_bests = np.roots([x_pw4_coeff, x_pw3_coeff, x_pw2_coeff, x_pw1_coeff, x_pw0_coeff])
    #if printDebugInfoToScreen: print(f"The four roots of x = {x_bests}")

    real_xBests = [np.real(x) for x in x_bests if np.isreal(x)]  # returns real part if number is real - discards the 0j part
    real_yBests = [np.polyval(curve, i) for i in real_xBests]  # possible y coordinates

    best_pts= [x for x in zip(real_xBests, real_yBests)]
    #if printDebugInfoToScreen: print(f"\nPossible locations of best points = {Pws}\n\n")

    # there are (usually) only 2 real points
    # there can be four real pints - a circle drawn with a give point as its center can cut a 2degree curve at most at 4 different points
    # but since our curve has a high eccentricity (away from circle e=0, towards a line e=inf), and our point is very close to the curve, that could not be the case

    bestPt = None

    if farPoint is not None:
        maxDist = -1
        for idx, pt in enumerate(best_pts):
            #if printDebugInfoToScreen: print(f"Point index = {idx}, pt = {pt}")
            x, y = pt
            dist_from_farPoint = np.sqrt((y - farPoint[1]) ** 2 + (x - farPoint[0]) ** 2)
            if dist_from_farPoint > maxDist:
                maxDist = dist_from_farPoint
                bestPt = (x, y)

    elif nearPoint is not None:
        minDist = 1e10
        for idx, pt in enumerate(best_pts):
            #if printDebugInfoToScreen: print(f"Point index = {idx}, pt = {pt}")
            x, y = pt
            dist_from_nearPoint = np.sqrt((y - nearPoint[1]) ** 2 + (x - nearPoint[0]) ** 2)
            if dist_from_nearPoint < minDist:
                minDist = dist_from_nearPoint
                bestPt = (x, y)

    else:
        logging.error(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Missing near point and far point. Specify at least one of them.")
        if printDebugInfoToScreen: print(f"ERROR: Missing near point and far point. Specify at least one of them.")

    return bestPt

def reflect_point_across_line(p1, p3, p2=None, m12=None, printDebugInfoToScreen=False):
    '''
    Reflects point p3 across line l12 formed by p1 and p2
    l12 can be specified either using p1 and p2 or with p1 and m12
    Returns reflected point p4.

    :param p1: (x1,y1) - coordinates of point p1
    :param p3: (x3,y3) - coordinates of point p3
    :param p2: (x2,y2) - coordinates of point p2
    :param m12: float value, slope of l12 = (y2-y1)/(x2-x1)
    :return: p4 (x4, y4)
    '''

    if p2 is None and m12 is None:
        if printDebugInfoToScreen: print(f"Error: Specifiy either slope of l12 or another point p2")
        logging.error(f"{os.path.basename(__file__)}>{inspect.stack()[0][3]}>> Specifiy either slope of l12 or another point p2")

    #l12 - line through p1 and p2
    x1, y1 = p1

    if p2 is not None: #calculate slope from second point on l12
        x2, y2 = p2
        m12 = (y2-y1)/(x2-x1) if (x2-x1) != 0 else (y2-y1)/(x2-x1+1e-10) #slope of l12

    #l34 - line through P0 and known point p3
    m34 = -1/(m12) if m12 != 0 else -1/(m12+1e-10) #slope of perpendicular

    #P0(x0,y0) = point of intersection of l12 and l34 - perpendicular projection of known point p3 on l12
    x3, y3 = p3
    x0 = (m12*x1 - m34*x3 + y3 - y1)/(m12-m34)
    y0 = m12*(x0 - x1) + y1
    if printDebugInfoToScreen: print(f"\nPerpendicular proj of p3 on l12 = ({x0},{y0})")

    x4 = 2*x0 - x3
    y4 = 2*y0 - y3

    p4 = (x4, y4)

    return p4

def run_interpolation_demo(printDebugInfoToScreen=True):
    '''
    Runs interpolation demo
    :return:
    '''

    #fullCowImg = cv2.imread("./frameSample_55710.jpg")  # full cow image
    fullCowImg = cv2.imread("./images/060922_2216_12.jpg")  # full cow image
    h, w, _ = fullCowImg.shape

    #instance_KPs = np.array(
    #    [[3.2905307e+02, 7.9342133e+02, 1.1288100e+01], [1.1160443e+02, 6.7395056e+02, 8.4008265e-01],
    #     [1.7640701e+02, 5.1561584e+02, 5.9272442e+00], [3.9529572e+02, 5.1705524e+02, 3.9843166e+00],
    #     [8.1723248e+02, 7.0561749e+02, 3.9593369e+01], [8.1723248e+02, 4.7099423e+02, 2.8083863e+00],
    #     [7.2506879e+02, 2.6228030e+02, 1.6525644e+01], [1.1700465e+03, 5.2137347e+02, 4.8193440e+00],
    #     [1.1858871e+03, 4.2781204e+02, 3.2210174e+00], [1.1470056e+03, 3.3281122e+02, 4.6865168e+00]])

    instance_KPs = np.array([[1322,  331,    2], [1495,  502,    2], [1426,  605,    2], [1264,  546,    2], [ 644,  507,    2], [ 739,  692,    2], [ 823,  869,    2], [ 386,  771,    2], [ 405,  849,    2], [ 449,  908,    2],], float) # new gt for images/060922_2216_12.jpg

    if printDebugInfoToScreen: print(f"Original KPs = \n{instance_KPs}\n")

    keypoint_names = kpn = ["left_shoulder", "withers", "right_shoulder", "center_back", "left_hip_bone",
                            "hip_connector", "right_hip_bone", "left_pin_bone", "tail_head", "right_pin_bone"]
    keypoint_connection_rules = [(kpn[1 - 1], kpn[4 - 1], (0, 0, 255)), (kpn[1 - 1], kpn[5 - 1], (0, 255, 0)),
                                 (kpn[1 - 1], kpn[2 - 1], (255, 0, 0)), (kpn[2 - 1], kpn[3 - 1], (255, 255, 0)),
                                 (kpn[2 - 1], kpn[4 - 1], (0, 255, 255)), (kpn[3 - 1], kpn[4 - 1], (0, 0, 0)),
                                 (kpn[3 - 1], kpn[7 - 1], (255, 255, 255)), (kpn[4 - 1], kpn[6 - 1], (255, 128, 128)),
                                 (kpn[4 - 1], kpn[5 - 1], (128, 255, 128)), (kpn[4 - 1], kpn[7 - 1], (128, 128, 255)),
                                 (kpn[5 - 1], kpn[6 - 1], (255, 255, 128)), (kpn[5 - 1], kpn[8 - 1], (255, 128, 255)),
                                 (kpn[6 - 1], kpn[7 - 1], (128, 255, 255)), (kpn[6 - 1], kpn[8 - 1], (255, 128, 64)),
                                 (kpn[6 - 1], kpn[10 - 1], (255, 64, 128)), (kpn[6 - 1], kpn[9 - 1], (128, 255, 64)),
                                 (kpn[7 - 1], kpn[10 - 1], (128, 64, 255)), (kpn[8 - 1], kpn[9 - 1], (64, 255, 128)),
                                 (kpn[9 - 1], kpn[10 - 1], (64, 128, 255))]

    # outImg = draw_and_connect_keypoints(img=fullCowImg, keypoints=instance_KPs, keypoint_threshold=0.5, keypoint_names=keypoint_names, keypoint_connection_rules=keypoint_connection_rules)

    # cv2.imshow('image', outImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # modifying instanceKP
    modified_KPs = instance_KPs.copy()
    # modified_KPs[keypoint_names.index('withers'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('left_shoulder'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('right_shoulder'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('center_back'), -1] = 0  # making it invisible

    # modified_KPs[keypoint_names.index('left_hip_bone'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('right_hip_bone'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('hip_connector'), -1] = 0  # making it invisible

    #modified_KPs[keypoint_names.index('left_pin_bone'),-1] = 0 #making it invisible
    modified_KPs[keypoint_names.index('left_pin_bone')] = [10,9,2]
    modified_KPs[keypoint_names.index('right_pin_bone'),-1] = 0 #making it invisible
    # modified_KPs[keypoint_names.index('tail_head'), -1] = 0  # making it invisible

    # modified_KPs[keypoint_names.index('left_hip_bone'), -1] = 0  # making it invisible
    # modified_KPs[keypoint_names.index('right_hip_bone'), -1] = 0  # making it invisible


    # modified_KPs[keypoint_names.index('left_hip_bone')] = [0,0,2]
    # modified_KPs[keypoint_names.index('right_hip_bone')] = [10,10,2]
    # modified_KPs[keypoint_names.index('tail_head')] = [11,-1,2]

    # modified_KPs[keypoint_names.index('hip_connector')] = [0, 0, 2]
    # modified_KPs[keypoint_names.index('tail_head')] = [10,10,2]


    if printDebugInfoToScreen: print(f"modified_KPs =\n {modified_KPs}\n")

    interpolated_KPs = interpolateKPs(instance_KPs=modified_KPs, keypoint_names=keypoint_names, keypoint_threshold=0.5, frameW=1920, frameH=1080)
    # interpolated_KPs, curve = interpolateKPs(instance_KPs=modified_KPs, keypoint_names=keypoint_names, keypoint_threshold=0.5)
    if printDebugInfoToScreen: print(f"interpolated_KPs = {interpolated_KPs}")

    outImg_interpolated = draw_and_connect_keypoints(img=fullCowImg, keypoints=interpolated_KPs, keypoint_threshold=0.5,
                                                     keypoint_names=keypoint_names,
                                                     keypoint_connection_rules=keypoint_connection_rules)
    # cv2.imshow('image', outImg_interpolated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(cv2.cvtColor(outImg_interpolated, cv2.COLOR_BGR2RGB))

    # x = np.linspace(0, w, 100)
    # y = [np.polyval(curve, i) for i in x]
    # plt.plot(x, y)

    plt.show()

class KP_ERROR_LOCALIZER():

    '''
    NOT USED!!!
    See experiment_getAffectedKPs.py for functions that are used.
    '''

    def __init__(self, dsetFilePath=None, img_dir=None, printDebugInfoToScreen=False):
        self.dsetFilePath = dsetFilePath
        if self.dsetFilePath is not None: #it can be none if you are using this class elsewhere, eg: in _autoCattloggerBase.py
            self.dsetName = self.dsetFilePath.split('/')[-1].split('.')[0]
        self.coco = COCO(self.dsetFilePath)
        self.img_dir = img_dir
        self.printDebugInfoToScreen = printDebugInfoToScreen

    def kp_error_localizer(self, kp_error_matrix, instance_KPs, keypoint_names, keypoint_threshold=0.5):
        '''
        To tell which of the 10 keypoints are detected in the wrong places.
        :param instance_KPs:
        :param keypoint_names:
        :param keypoint_threshold:
        :return:
        '''

        pass

    def build_kpRuleFails_to_kp_map(self, instance_KPs, keypoint_names, keypoint_threshold=0.5, frameW=1920, frameH=1080):
        '''
        Builds a map of the kp rules that are failed to the kp that is wrongly detected - aggregates from the whole dataset.
        :param instance_KPs:
        :param keypoint_names:
        :param keypoint_threshold:
        :return: kp_error_matrix?
        '''

        cats = self.coco.loadCats(self.coco.getCatIds())
        catIds = self.coco.getCatIds(catNms=[cats[0]])  # does not work
        cat_ids = self.coco.getCatIds()  # works

        #CHANGE LATER
        #take all images later
        imgIds = self.coco.getImgIds(catIds=catIds)[10]

        if self.printDebugInfoToScreen: print(f"\nThere are a total of {len(imgIds)} annotated images")
        if self.printDebugInfoToScreen: print(f"CatIds = {catIds}, cats = {cats}, \nimgIds = {imgIds}")

        keypoint_names = kpn = cats[0]['keypoints']
        # kp connection rules is also in cats[0]['skeleton'] but it has no color info
        keypoint_connection_rules = keypoint_connection_rules = [(kpn[1 - 1], kpn[4 - 1], (0, 0, 255)), (kpn[1 - 1], kpn[5 - 1], (0, 255, 0)), (kpn[1 - 1], kpn[2 - 1], (255, 0, 0)), (kpn[2 - 1], kpn[3 - 1], (255, 255, 0)), (kpn[2 - 1], kpn[4 - 1], (0, 255, 255)), (kpn[3 - 1], kpn[4 - 1], (0, 0, 0)), (kpn[3 - 1], kpn[7 - 1], (255, 255, 255)), (kpn[4 - 1], kpn[6 - 1], (255, 128, 128)), (kpn[4 - 1], kpn[5 - 1], (128, 255, 128)), (kpn[4 - 1], kpn[7 - 1], (128, 128, 255)), (kpn[5 - 1], kpn[6 - 1], (255, 255, 128)), (kpn[5 - 1], kpn[8 - 1], (255, 128, 255)), (kpn[6 - 1], kpn[7 - 1], (128, 255, 255)), (kpn[6 - 1], kpn[8 - 1], (255, 128, 64)), (kpn[6 - 1], kpn[10 - 1], (255, 64, 128)), (kpn[6 - 1], kpn[9 - 1], (128, 255, 64)), (kpn[7 - 1], kpn[10 - 1], (128, 64, 255)), (kpn[8 - 1], kpn[9 - 1], (64, 255, 128)), (kpn[9 - 1], kpn[10 - 1], (64, 128, 255))]

        for imgId in imgIds:

            # imgPath = imgDir + '/' + coco.imgs[imgId]['path'].split('/')[-1]
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=cat_ids[0], iscrowd=None)
            annos = self.coco.loadAnns(annIds)  # there is only one annotation
            print(f"annotations of the given category = {annos}")

            # mask = coco.annToMask(annos[0])  # bianry mask with values in [0,1], len of annos is just 1 - there is only one annotated cow per image

            for anno in annos:
                mask = self.coco.annToMask(
                    anno)  # bianry mask with values in [0,1] #mask is needed to compute cowW and cowH (length)
                mask = mask.astype(np.uint8) * 255
                cowW, cowH = self.get_cowW_cowH(mask_img=mask)

                keypoints = np.array(anno['keypoints'], int).reshape((-1, 3))
                if self.printDebugInfoToScreen: print(f"Keypoints = {keypoints}")

    def generate_kpRulesFails_kp_matrix_perInstance(self,instance_KPs, keypoint_names, keypoint_threshold=0.5, frameW=1920, frameH=1080):
        '''
                Builds a map of the kp rules that are failed to the kp that is wrongly detected.
                :param instance_KPs:
                :param keypoint_names:
                :param keypoint_threshold:
                :return: kp_error_matrix?
                '''

        '''
        LOGIC:
        We start with a human annotated image and vary the locations of keypoints one at a time.
        We will move each keypoint to random locations away from the original location and record the errors in the dictionary of errors/ a list.
        Having done the same each keypoint we see if the mapping is one-to-one and try to invert the mapping.
        This will help us localize the error to the keypoint that is wrongly detected. 
        We could then throw away this keypoint and interpolate one in its place.
        We could later add this image with automatically corrected keypoints to our training set and improve kp recognition training - full self supervision.
        '''

        # check visibility
        # ask if you need all visible
        # iterate over each kp
        #   iterate over differrent Radii bins
        #       generate uniformly random theta and radius (within the bin) - convert to x y coordinates
        #       use this point as new kp location -> find error vector
        #       create error vec -> wrong keypoint dictionary
        # if more than one value exists for a key, then each value has uniform probability?
        # or count the number of times it each kp occurs for each error vec {errorVec: (kp, #occurrences)} and then compute a prob ditribution for each error vector.
        # come up with how you would use this prob mapping later.

        pass


if __name__ == "__main__":
    from helper_for_infer import draw_and_connect_keypoints

    run_interpolation_demo()

    #try to put all interpolation functions under an interpolation class later.
