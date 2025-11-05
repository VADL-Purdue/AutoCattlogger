'''
Author: Manu Ramesh
Contains functions to help reading video and depth data from HDF5 files.
The orbbec camera saves the video and depth data in HDF5 format.
'''

import h5py, cv2, numpy as np

class VideoCapture_HDF5():

    '''
    I am using a custom format for saving data into the HDF4 files.
    Here i will be writing functions that read data stored in this format.
    '''

    def __init__(self, filePath, startIdx=0):

        self.filePath = filePath
        with h5py.File(filePath, 'r') as hdf:
            self.keys = list(hdf.keys())
            self.nFrames = len(hdf['timestamps'])

            # save frame width and height info
            # color image
            colorImgJpg = hdf['color_images_jpg'][0] 
            colorImg = cv2.imdecode(np.frombuffer(colorImgJpg, np.uint8), cv2.IMREAD_COLOR)
            self.frame_height, self.frame_width = colorImg.shape[:2]
        
        assert 0 <= startIdx, "Start index should be >= 0"
        assert startIdx < self.nFrames, "Start index should be < number of frames in the file"
        self.index = startIdx

        
    def read(self,):
        #just read the color image
        #compatible with OpenCV style
        with h5py.File(self.filePath, 'r') as hdf:

            if self.index >= self.nFrames:
                # Format: return <status>, <colorImg>, <depthImg>, <ts>, <tsStr>
                return False, None


            # color image
            colorImgJpg = hdf['color_images_jpg'][self.index] 
            colorImg = cv2.imdecode(np.frombuffer(colorImgJpg, np.uint8), cv2.IMREAD_COLOR)
            

            self.index += 1

            return True, colorImg



    def read_all(self, get8bitDepth=False):
        #read all info, return all info
        #useful when you need to do more
        with h5py.File(self.filePath, 'r') as hdf:

            if self.index >= self.nFrames:
                # Format: return <status>, <colorImg>, <depthImg>, <ts>, <tsStr>
                return False, None, None, None, None

            ts = hdf['timestamps'][self.index]
            tsStr = hdf['time_strings'][self.index].decode('utf-8')

            # color image
            colorImgJpg = hdf['color_images_jpg'][self.index] 
            colorImg = cv2.imdecode(np.frombuffer(colorImgJpg, np.uint8), cv2.IMREAD_COLOR)
            

            # depth image
            depthImg = hdf['depth_images_png'][self.index]
            depthImg = cv2.imdecode(np.frombuffer(depthImg, np.uint8), cv2.IMREAD_UNCHANGED) #has 16 bit depth

            if get8bitDepth:
                # Convert depth image to 8-bit for display
                depthImg = cv2.normalize(depthImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depthImg = cv2.applyColorMap(depthImg, cv2.COLORMAP_JET)


            self.index += 1

            return True, colorImg, depthImg, ts, tsStr
        
    def read_all_atIdx(self, get8bitDepth_also=False, idx=None):

        #read all info at the supplied index, return all info
        #useful when you need to do more
        with h5py.File(self.filePath, 'r') as hdf:

            if idx >= self.nFrames:
                # Format: return <status>, <colorImg>, <depthImg>, <ts>, <tsStr>
                return False, None, None, None, None

            ts = hdf['timestamps'][idx]
            tsStr = hdf['time_strings'][idx].decode('utf-8')

            # color image
            colorImgJpg = hdf['color_images_jpg'][idx] 
            colorImg = cv2.imdecode(np.frombuffer(colorImgJpg, np.uint8), cv2.IMREAD_COLOR)
            

            # depth image
            depthImg = hdf['depth_images_png'][idx]
            depthImg = cv2.imdecode(np.frombuffer(depthImg, np.uint8), cv2.IMREAD_UNCHANGED) #has 16 bit depth

            if get8bitDepth_also:
                # Convert depth image to 8-bit for display
                depthImg8bit = cv2.normalize(depthImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depthImg8bit = cv2.applyColorMap(depthImg8bit, cv2.COLORMAP_SPRING)
                return True, colorImg, depthImg, depthImg8bit, ts, tsStr

            ## self.index += 1 # do not increment self.index

            return True, colorImg, depthImg, ts, tsStr
        
    def get(self, propId):
        if propId == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return self.nFrames
        elif propId == cv2.CAP_PROP_FPS:
            return 15  # Default FPS
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frame_width
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height
        else:
            print(f"Property id {propId} not implemented")
            return -1
        
    #we do not need a close fn
    def release(self,):
        pass

    #can add fns to read at index, read reverse, etc.