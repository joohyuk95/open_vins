#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from random import gauss, uniform

class IMUNoiseNode:
    def __init__(self):
        rospy.init_node('imu_noise_node')
        self.imu_pub = rospy.Publisher('/imu1', Imu, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu0', Imu, self.imu_callback)
        self.rate_noise_stddev = rospy.get_param('~rate_noise_stddev', 0.0)
        self.rate_bias_mean = rospy.get_param('~rate_bias_mean', 0.0)
        self.rate_bias_stddev = rospy.get_param('~rate_bias_stddev', 0.0)
        self.accel_noise_stddev = rospy.get_param('~accel_noise_stddev', 0.0)
        self.accel_bias_mean = rospy.get_param('~accel_bias_mean', 0.0)
        self.accel_bias_stddev = rospy.get_param('~accel_bias_stddev', 0.0)

    def imu_callback(self, imu_msg):
        str = "Noise enabled -------------"
        rospy.loginfo(str)
        str = "rate_noise_stddev - %f"%self.rate_noise_stddev
        rospy.loginfo(str)

        str = "rate_bias_mean - %f"%self.rate_bias_mean
        rospy.loginfo(str)
        
        str = "rate_bias_stddev - %f"%self.rate_bias_stddev
        rospy.loginfo(str)
        
        str = "accel_noise_stddev - %f"%self.accel_noise_stddev
        rospy.loginfo(str)
        
        str = "accel_bias_mean - %f"%self.accel_bias_mean
        rospy.loginfo(str)
        
        str = "accel_bias_stddev - %f"%self.accel_bias_stddev
        rospy.loginfo(str)
        
        str = "-------------------------------"

        imu_with_noise = Imu()
        imu_with_noise.header = imu_msg.header
        imu_with_noise.orientation = imu_msg.orientation

        # Add noise and bias to angular rates
        imu_with_noise.angular_velocity.x = imu_msg.angular_velocity.x + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev)
        imu_with_noise.angular_velocity.y = imu_msg.angular_velocity.y + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev)
        imu_with_noise.angular_velocity.z = imu_msg.angular_velocity.z + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev)

        # Add noise and bias to linear accelerations
        imu_with_noise.linear_acceleration.x = imu_msg.linear_acceleration.x + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev)
        imu_with_noise.linear_acceleration.y = imu_msg.linear_acceleration.y + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev)
        imu_with_noise.linear_acceleration.z = imu_msg.linear_acceleration.z + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev)

        self.imu_pub.publish(imu_with_noise)

    def _sample_bias(self, bias_mean, bias_stddev):
        bias = gauss(bias_mean, bias_stddev)
        if uniform(0, 1) < 0.5:
            bias *= -1
        return bias

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    imu_noise_node = IMUNoiseNode()
    imu_noise_node.run()
