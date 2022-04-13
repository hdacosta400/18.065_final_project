'''
Runs through vids in posture_videos and extracts ONLY the skeleton

to be used for alternative ML model
'''

import time
import cv2
import os 
import sys
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

def get_skeleton(img, models_path, blending=False):
    '''
    Runs an image through openpose and extracts the body keypoints

    models_path: path to models folder in openpose
    blending: only render the skeleton

    Return dictionary mapping body key points to 2d coordinates

    '''
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    
    params["model_folder"] = models_path
    params["disable_blending"] = blending # only render skeleton
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(img) # or do it on a per-image basis as shown here
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))


    cv2.imwrite(img, datum.cvOutputData) # save skeleton images

class ExtractPostureFromVideo:
    '''
    Uses ExtractFeatures as black box to extract featuers from images sampled from video, and writes to csv for ML training
    '''
    def __init__(self, video_filename, good_posture, frame_rate):
        '''
        video_filename: a video of the posture
        good_posture: is the vid good posture
        '''
        self.video = video_filename
        self.image_filenames = []
        self.good_posture = 1 if good_posture else 0
        self.frame_rate = frame_rate

    '''
    Convert posture video into series of images
    '''
    def get_image_frames(self):
        vidcap = cv2.VideoCapture(self.video)
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                img_filename = "skeleton_images/image_{}_{}.jpg".format(time.time(), self.good_posture)
                self.image_filenames.append(img_filename)
                cv2.imwrite(img_filename, image)     # save frame as JPG file
            return hasFrames
        sec = 0
        counter = 1
        success = getFrame(sec)
        while success:
            counter += 1
            sec = sec + self.frame_rate
            sec = round(sec, 2)
            success = getFrame(sec)
    
    '''
    Call OpenPose and extract skeleton features, put skeleton data into CV file for processing
    '''
    def get_posture_skeletons(self):
        for img in self.image_filenames:
            get_skeleton(img, "../openpose/models/", True)



    '''
    Wrapper to call all functions
    '''
    def extract_posture_images(self):
        self.get_image_frames()
        self.get_posture_skeletons()

if __name__ == "__main__":
    for filename in os.listdir('./posture_videos'):
        if filename.endswith('.mov'):
            img_filename = os.path.join('./posture_videos', filename)
            print("video:", img_filename)
            posture,_,_ = filename.split('_') # get good / bad out of name
            good_posture = 1 if posture == 'good' else 0
            v = ExtractPostureFromVideo(img_filename, good_posture, .2)
            v.extract_posture_images()
            


    pass 