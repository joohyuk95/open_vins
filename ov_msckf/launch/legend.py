#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node("trajectory_labels")

pub_imm = rospy.Publisher("/imm_vio_label", String, queue_size=1)
pub_imu1 = rospy.Publisher("/imu1_label", String, queue_size=1)
pub_imu2 = rospy.Publisher("/imu2_label", String, queue_size=1)

rate = rospy.Rate(1)
while not rospy.is_shutdown():
    pub_imm.publish("IMM-VIO")
    pub_imu1.publish("IMU1")
    pub_imu2.publish("IMU2")
    rate.sleep()