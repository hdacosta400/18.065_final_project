import cv2
from enum import Enum
import os
from sys import platform
import argparse
import sys
import csv
import numpy as np

sys.path.append('/usr/local/python')
from openpose import pyopenpose as op
'''
 HOPEFULLY this just works if you run the following commands:
    1) cd into openpose/build and run 'sudo make install' (or Windows equivalent)
    2) cd into openpose/build/python/openpose and run 'sudo make install'

    This should put the openpose library in your default python path, allowing it to be used
    like any other import 
        you can run 'sudo make uninstall' in these directories as well if you want to remove it

If not (very bad!), then go to openpose instructions:
    - you might have to install cmake and re build the binaries for the library
        guide - https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#prerequisites
    - you may get errors along the lines of these forums:
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1143
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1740
        https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1623
        https://stackoverflow.com/questions/54297627/qt-could-not-find-the-platform-plugin-cocoa
    I'm not sure exactly what fixed it, but follow these forms and try to re-configure the openpose library using the 
    cmake-gui a bunch of times and eventually (hopefully) it'll work

    DO NOT PUSH TO MASTER if you can't get this working lol

Might be helpful:
    https://robinreni96.github.io/computervision/Python-Openpose-Installation/  
    https://medium.com/@erica.z.zheng/installing-openpose-on-ubuntu-18-04-cuda-10-ebb371cf3442

optimally we can all have this working on our own machines, if not send me (Howard) data that you want 
to train/test/validate
'''

class PostureType(Enum):
    GOOD = 1
    BAD = 2 


def get_skeleton(img, models_path):
    '''
    Runs an image through openpose and extracts the body keypoints

    models_path: path to models folder in openpose

    Return dictionary mapping body key points to 2d coordinates
    '''
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    
    params["model_folder"] = models_path

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(img) # or do it on a per-image basis as shown here
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    raw_data_points = datum.poseKeypoints[0]
    # more on how datum is formatted:
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#keypoint-format-in-datum-advanced
    # print("Body keypoints: \n" + str(raw_data_points), len(raw_data_points))
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData) # display images

    # if we decide we want images in the future
    cv2.imwrite(img, datum.cvOutputData) # save skeleton images

    keypoints_for_posture = [0, 1, 2 ,5, 15, 16, 17, 18]
            # indices for  nose,neck,Rshoulder,Lshoulder,Reye, Leye, Rear, Lear
    data_points = [] # list of cartesian points in space for each of the body parts

    for p in keypoints_for_posture:
        point = raw_data_points[p]
        data_points.append((point[0], point[1]))
    # print("extracted body coordinates:" , data_points)

    data_dictionary = {}
    # current body parts we are examining for posture
    BODY_PARTS = ["nose","neck","r_shoulder","l_shoulder","r_eye", "l_eye", "r_ear", "l_ear"]

    for i in range(len(BODY_PARTS)):
        data_dictionary[BODY_PARTS[i]] = np.array(data_points[i])
    
    return data_dictionary


class ExtractFeatures:
    '''
    Takes one image, runs it through openpose and extracts features from it
    Extracts the features from one skeletonized image
    '''
    def get_vector(self, p1, p2):
        '''
        Gets the vector spanning points p1 and p2
        '''
        return np.array( [ abs(p1[0] - p2[0]), abs(p1[1] - p2[1]) ] )

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def extract_features_from_body_points(self, data_dictionary):
        '''
        From 8 body coordinates extracted from OpenPose, caluclate features from them (to be fed into NN)

        Current features:
            - Shoulder to shoulder distance 
            - nose to neck distance
            - Shoulder - neck angles
            - Ear to ear alignment 

        data_dictionary: dict mapping key body parts to coordinate positions in image

        returns values for features in order

        '''
        shoulder_to_shoulder_dist = np.linalg.norm(data_dictionary["l_shoulder"] - data_dictionary["r_shoulder"])
        nose_to_neck_dist = np.linalg.norm(data_dictionary["nose"] - data_dictionary["neck"])

        
        neck_vector = self.get_vector(data_dictionary["neck"], data_dictionary["nose"])

        r_shoulder_vector =  self.get_vector(data_dictionary["r_shoulder"], data_dictionary["neck"])
        l_shoulder_vector =  self.get_vector(data_dictionary["l_shoulder"], data_dictionary["neck"])

        # angle formed between shoulders and neck
        r_shoulder_angle = self.angle_between(r_shoulder_vector, neck_vector)
        l_shoulder_angle = self.angle_between(l_shoulder_vector, neck_vector)

        ear_vector = self.get_vector(data_dictionary["r_ear"], data_dictionary["l_ear"])

        # can determine horizontal alignment of ears by finding angle between ear_vector and x_axis
        e2e_alignment = self.angle_between(ear_vector, np.array([1,0]))

        return [shoulder_to_shoulder_dist, nose_to_neck_dist, r_shoulder_angle, l_shoulder_angle, e2e_alignment]


class ExtractPostureFromVideo:
    '''
    Uses ExtractFeatures as black box to extract featuers from images sampled from video, and writes to csv for ML training
    '''
    def __init__(self, video_filename, posture_type, frame_rate):
        '''
        video_filename: a video of the posture
        posture_type: PostureType.GOOD or PostureType.BAD, to be used for labeling
        images
        frame_rate: the rate at which the video will be sampled for images
                    (1 image per frame_rate seconds)
        '''
        self.video = video_filename
        self.type = posture_type
        self.image_filenames = []
        self.frame_rate = frame_rate
        # list of posture_features for image
        self.feature_extractor = ExtractFeatures()

    '''
    Convert posture video into series of images
    '''
    def get_image_frames(self):
        vidcap = cv2.VideoCapture(self.video)
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                img_filename = "posture_images/image"+str(counter)+".jpg"
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
    def get_posture_features(self):
        '''
        Returns dictionary mapping body parts to their coordinates

        In order to add more body parts:
            find corresponding index in bodypose output
            add index to keypoints_index list
            add name of body part to body_parts list
        '''
        for img in self.image_filenames:

            data_dictionary = get_skeleton(img, "../openpose/models/")
            posture_features = self.feature_extractor.extract_features_from_body_points(data_dictionary)
            #label for the data extracted from the video
            good_posture = 1 if self.type == PostureType.GOOD else 0

            # csv will have  6 cols = 5 posture feature inputs and 1 output (good (1) or bad (0) posture)
            # ordered as such:
                # [shoulder_to_shoulder_dist, nose_to_neck_dist, r_shoulder_angle, l_shoulder_angle, e2e_alignment, good_posture]
            # with open('training_data.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter=',',
            #             quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #     writer.writerow(posture_features + [good_posture])
    '''
    Wrapper to call all functions
    '''
    def extract_posture_data(self):
        self.get_image_frames()
        self.get_posture_features()

        

if __name__ == "__main__":
    # if you want to add a new feature, need to recreate entire csv (unless you find a way to add a column, maybe pandas or something idk)
    # NUM_GOOD_POSTURE_VIDS = 3
    # NUM_BAD_POSTURE_VIDS = 1
    # print("processing bad posture vids.....")
    # for i in range(1, NUM_BAD_POSTURE_VIDS + 1):
    #     print("processing ./posture_videos/bad_posture_0{}.mov".format(i))
    #     e = ExtractPosture("./posture_videos/bad_posture_0{}.mov".format(i), PostureType.BAD, .5)
    #     e.extract_posture_data()
    #     print("done!")

    # print("processing good posture_vids.....")
    # for i in range(1, NUM_GOOD_POSTURE_VIDS + 1):
    #     print("processing ./posture_videos/good_posture_0{}.mov".format(i))
    #     e = ExtractPosture("./posture_videos/good_posture_0{}.mov".format(i), PostureType.GOOD, .5)
    #     e.extract_posture_data()
    #     print("done!")


    # otherwise, follow naming convention and just run on the specific file e.g. 
    # e = ExtractPostureFromVideo("./posture_videos/bad_posture_03.mov", PostureType.BAD, .2)
    # e.extract_posture_data() 

    # e = ExtractPostureFromVideo("./posture_videos/good_posture_04.mov", PostureType.GOOD, .2)
    # e.extract_posture_data() 
    # pass
    img = "/Users/marco/Desktop/pic.jpg"
    path='../../openpose/models/'
    get_skeleton(img, path)