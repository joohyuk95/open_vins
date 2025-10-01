#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
import cv2
import numpy as np
import ros_numpy

class DisparityGenerator:

    def __init__(self):
        rospy.init_node('disparity_generator')

        # Set up the subscribers
        self.left_sub = rospy.Subscriber('/stereo/left/image_raw', Image, self.left_callback)
        self.right_sub = rospy.Subscriber('/stereo/right/image_raw', Image, self.right_callback)
        self.info_sub = rospy.Subscriber('/stereo/left/camera_info', CameraInfo, self.info_callback)

        # Set up the publisher
        self.disparity_pub = rospy.Publisher('/disparity', Image, queue_size=1)

        # Initialize the variables
        self.left_img = None
        self.right_img = None
        self.calibration = None
        #self.bridge = CvBridge()

    def left_callback(self, data):
        self.left_img = self.image_msg_to_cv2(data)
        self.image_header = data.header

    def right_callback(self, data):
        self.right_img = self.image_msg_to_cv2(data)

    def info_callback(self, data):
        self.calibration = (data.K[0], data.K[4], data.P[3])

    def image_msg_to_cv2(self, data):
        cv_image1 = ros_numpy.numpify(data)
        cv_image = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2RGB)
        return cv_image

    def generate_disparity(self):
        if self.left_img is not None and self.right_img is not None and self.calibration is not None:
            stereo = cv2.StereoSGBM_create(minDisparity=-64, numDisparities=192, blockSize=11,
                                            P1=8 * 3 * 11 ** 2, P2=32 * 3 * 11 ** 2, disp12MaxDiff=1,
                                            uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
            disparity = stereo.compute(self.left_img, self.right_img)
            disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            disparity = cv2.convertScaleAbs(disparity)
            #disparity_msg = self.bridge.cv2_to_imgmsg(disparity, encoding="mono8")
            try:
                disparity_msg = Image(encoding="mono8")
                disparity_msg.header= self.image_header #self.imghead
                # Fill the image data 
                disparity_msg.height, disparity_msg.width = disparity.shape
                disparity_msg.data = disparity.ravel().tobytes() # or .tostring()
                #ros_image.data = cv_image.tostring()
                disparity_msg.step=disparity_msg.width
                disparity_msg.encoding = "mono8"
                self.disparity_pub.publish(disparity_msg)
            except Exception as e: print(e)
            
            #cv2.imshow('Disparity Image', disparity)
            #cv2.waitKey(0.1)

    def run(self):
        while not rospy.is_shutdown():
            self.generate_disparity()

if __name__ == '__main__':
    try:
        disparity_generator = DisparityGenerator()
        disparity_generator.run()
    except rospy.ROSInterruptException:
        pass