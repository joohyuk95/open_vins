#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64, Bool
import sys, select, termios, tty
from sensor_msgs.msg import Imu
from random import gauss, uniform

class IMUNoiseParamControlNode:
    def __init__(self):
        rospy.init_node('imu_noise_param_control_node')
        self.rate_noise_stddev = rospy.get_param('~rate_noise_stddev', 0.0)
        self.rate_bias_mean = rospy.get_param('~rate_bias_mean', 0.0)
        self.rate_bias_stddev = rospy.get_param('~rate_bias_stddev', 0.0)
        self.accel_noise_stddev = rospy.get_param('~accel_noise_stddev', 0.0)
        self.accel_bias_mean = rospy.get_param('~accel_bias_mean', 0.0)
        self.accel_bias_stddev = rospy.get_param('~accel_bias_stddev', 0.0)
        self.enable_noise = True
        self.restore_values = {
            'rate_noise_stddev': self.rate_noise_stddev,
            'rate_bias_mean': self.rate_bias_mean,
            'rate_bias_stddev': self.rate_bias_stddev,
            'accel_noise_stddev': self.accel_noise_stddev,
            'accel_bias_mean': self.accel_bias_mean,
            'accel_bias_stddev': self.accel_bias_stddev
        }
        #self.param_pub = rospy.Publisher('/imu_noise_params', Float64, queue_size=1)
        self.enable_pub = rospy.Publisher('/enable_noise', Bool, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu1', Imu, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu0', Imu, self.imu_callback)

    def restore_param_values(self):
        self.rate_noise_stddev = self.restore_values['rate_noise_stddev']
        self.rate_bias_mean = self.restore_values['rate_bias_mean']
        self.rate_bias_stddev = self.restore_values['rate_bias_stddev']
        self.accel_noise_stddev = self.restore_values['accel_noise_stddev']
        self.accel_bias_mean = self.restore_values['accel_bias_mean']
        self.accel_bias_stddev = self.restore_values['accel_bias_stddev']
        self.publish_param_values()

    def delault_param_values(self):
        self.restore_values['rate_noise_stddev'] = self.rate_noise_stddev
        self.restore_values['rate_bias_mean'] = self.rate_bias_mean
        self.restore_values['rate_bias_stddev'] = self.rate_bias_stddev
        self.restore_values['accel_noise_stddev'] = self.accel_noise_stddev
        self.restore_values['accel_bias_mean'] = self.accel_bias_mean
        self.restore_values['accel_bias_stddev'] = self.accel_bias_stddev
        self.rate_noise_stddev = 0.0
        self.rate_bias_mean = 0.0
        self.rate_bias_stddev = 0.0
        self.accel_noise_stddev = 0.0
        self.accel_bias_mean = 0.0
        self.accel_bias_stddev = 0.0
        self.publish_param_values()

    def publish_param_values(self):
        param_values = Float64()
        param_values.data = [
            self.rate_noise_stddev,
            self.rate_bias_mean,
            self.rate_bias_stddev,
            self.accel_noise_stddev,
            self.accel_bias_mean,
            self.accel_bias_stddev
        ]
        rospy.set_param('~rate_noise_stddev', self.rate_noise_stddev)
        rospy.set_param('~rate_bias_mean', self.rate_bias_mean)
        rospy.set_param('~rate_bias_stddev', self.rate_bias_stddev)
        rospy.set_param('~accel_noise_stddev', self.accel_noise_stddev)
        rospy.set_param('~accel_bias_mean', self.accel_bias_mean)
        rospy.set_param('~accel_bias_stddev', self.accel_bias_stddev)

    def enable_disable_noise(self):
        self.enable_noise = not self.enable_noise
        enable_msg = Bool()
        enable_msg.data = self.enable_noise
        self.enable_pub.publish(enable_msg)

    def restore_terminal_settings(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def imu_callback(self, imu_msg):

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

    def handle_keyboard_input(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.rate_noise_stddev += 0.0024
                        str = "rate_noise_stddev - %f"%self.rate_noise_stddev
                        rospy.loginfo(str)
                    elif key == 'a':
                        self.rate_noise_stddev = max(0.0, self.rate_noise_stddev - 0.0024)
                        str = "rate_noise_stddev - %f"%self.rate_noise_stddev
                        rospy.loginfo(str)
                    elif key == 'w':
                        self.rate_bias_mean += 0.02
                        str = "rate_bias_mean - %f"%self.rate_bias_mean
                        rospy.loginfo(str)
                    elif key == 's':
                        self.rate_bias_mean = max(0.0, self.rate_bias_mean - 0.02)
                        str = "rate_bias_mean - %f"%self.rate_bias_mean
                        rospy.loginfo(str)
                    elif key == 'e':
                        self.rate_bias_stddev += 0.02
                        str = "rate_bias_stddev - %f"%self.rate_bias_stddev
                        rospy.loginfo(str)
                    elif key == 'd':
                        self.rate_bias_stddev = max(0.0, self.rate_bias_stddev - 0.02)
                        str = "rate_bias_stddev - %f"%self.rate_bias_stddev
                        rospy.loginfo(str)
                    elif key == 'r':
                        self.accel_noise_stddev += 0.0283
                        str = "accel_noise_stddev - %f"%self.accel_noise_stddev
                        rospy.loginfo(str)
                    elif key == 'f':
                        self.accel_noise_stddev = max(0.0, self.accel_noise_stddev - 0.0283)
                        str = "accel_noise_stddev - %f"%self.accel_noise_stddev
                        rospy.loginfo(str)
                    elif key == 't':
                        self.accel_bias_mean += 0.02
                        str = "accel_bias_mean - %f"%self.accel_bias_mean
                        rospy.loginfo(str)
                    elif key == 'g':
                        self.accel_bias_mean = max(0.0, self.accel_bias_mean - 0.02)
                        str = "accel_bias_mean - %f"%self.accel_bias_mean
                        rospy.loginfo(str)
                    elif key == 'y':
                        self.accel_bias_stddev += 0.02
                        str = "accel_bias_stddev - %f"%self.accel_bias_stddev
                        rospy.loginfo(str)
                    elif key == 'h':
                        self.accel_bias_stddev = max(0.0, self.accel_bias_stddev - 0.02)
                        str = "accel_bias_stddev - %f"%self.accel_bias_stddev
                        rospy.loginfo(str)
                    elif key == ' ':
                        if not self.enable_noise:
                            self.enable_disable_noise()
                            self.restore_param_values()
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
                            rospy.loginfo(str)
                        else:
                            self.enable_disable_noise()
                            self.delault_param_values()
                            str = "Noise disabled"
                            rospy.loginfo(str)
                    elif key == '\x03':
                        break
                    self.publish_param_values()
                rate.sleep()
        finally:
            self.restore_terminal_settings()

if __name__ == '__main__':
    imu_noise_param_control_node = IMUNoiseParamControlNode()
    imu_noise_param_control_node.handle_keyboard_input()
