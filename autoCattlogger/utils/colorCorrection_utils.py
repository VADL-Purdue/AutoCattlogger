'''
Author: Manu Ramesh

File with less often used utility functions.
'''

from matplotlib import pyplot as plt
import numpy as np, cv2
import pdb


def findDiffImage(img1, img2):
    '''
    Some helper function. 
    I used this to find the difference between two black point images. (With and without sRGB to linear conversion.)
    To find the difference between two (Extended) blackpoint images
    :param img1:
    :param img2:
    :return:
    '''

    img1F = img1#/255.0
    img2F = img2#/255.0
    diffImg = abs(img1F - img2F)

    plt.imshow(diffImg)
    plt.show()

def plotGammaCurves():
    '''
    Some function to plot the gamma curves for sRGB to linear and linear to sRGB.
    To check if the gamma curves are correct.
    https://github.com/PetterS/opencv_srgb_gamma
    :return:    
    '''


    def to_linear(srgb_float):

        #linear = np.float32(srgb) / 255.0
        linear = srgb_float.copy()

        less = linear <= 0.04045
        linear[less] = linear[less] / 12.92
        linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
        return linear

    def from_linear(linear):
        srgb = linear.copy()
        less = linear <= 0.0031308
        srgb[less] = linear[less] * 12.92
        srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055
        return srgb # * 255.0


    #x = np.arange(256)
    x = np.linspace(0,1,101)
    y = to_linear(x)

    z = from_linear(x)

    w = from_linear(to_linear(x))

    plt.plot(x,y, label='sRGB to Linear') # T-Fun (EOTF)')           #Electrical-Optical Transfer Function
    plt.plot(x,z, label='Linear to sRGB') # T-Fun (OETF)')           #Optical-Electrical Transfer Function
    # plt.plot(x,w, label='sRGB to linear to sRGB') # TFun (OOTF)')    #Optical-Optical Transfer Function
    plt.legend()
    plt.title('Gamma Curves (sRGB to Linear and Linear to sRGB)')
    plt.xlabel('Input Intensity (sRGB or Linear)')
    plt.ylabel('Output Intensity (Linear or sRGB)')
    plt.show()


def findDiffVideo(vidPath1, vidPath2):
    '''
    To find the difference in black point corrected videos using different black point corrected images.
    We study the effect after thresholding.

    :param vidPath1: path to original video
    :param vidPath2: path to blackpoint corrected video
    :return:
    '''

    cap1 = cv2.VideoCapture(vidPath1)
    cap2 = cv2.VideoCapture(vidPath2)

    ret1 = True; ret2 = True
    frameCount = 0


    while ret1 and ret2:

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 is False or ret2 is False:
            print("Breaking out since ret1 or ret2 is False")
            break

        frameCount+=1

        if frameCount < 3*30:
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
        #_, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY) #thresh of 127 doesnt work well here

        #pdb.set_trace()

        ogGreaterFrame = np.where(thresh1>thresh2, thresh1, gray1)
        bptCorrGreaterFrame = np.where(thresh2>thresh1, thresh2, gray2)


        ogGreaterFrame = cv2.resize(ogGreaterFrame, (640, 360)) # (1280, 720))
        bptCorrGreaterFrame = cv2.resize(bptCorrGreaterFrame,  (640, 360)) # (1280, 720))

        #cv2.imshow('Original', cv2.resize(frame1, (640, 360)))
        #cv2.imshow('Direct thresh greater than blkPt', ogGreaterFrame)
        #cv2.imshow('Blk pt corrected frame greater than direct thresholding', bptCorrGreaterFrame)

        #cv2.imshow('Direct thresh', cv2.resize(thresh1, (640, 360)))
        #cv2.imshow('Blk Pt Corrected Thresh', cv2.resize(thresh2, (640, 360)))
        comboImg = np.vstack((cv2.resize(gray1, (640, 360)), cv2.resize(thresh1, (640, 360)), cv2.resize(thresh2, (640, 360))))
        cv2.imshow('Original Frame grayscaled above, Direct thresh middle vs blk pt thresh below', comboImg)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    cap1.release()
    cap2.release()