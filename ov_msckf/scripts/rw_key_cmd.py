#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64, Bool
import sys, select, termios, tty
from sensor_msgs.msg import Imu
from random import gauss, uniform
import numpy as np

class IMUNoiseParamControlNode:
    def __init__(self):
        rospy.init_node('imu_noise_param_control_node')

        # --- Gyro parameters ---
        self.rate_noise_stddev = rospy.get_param('~rate_noise_stddev', 0.0)
        self.rate_bias_mean = rospy.get_param('~rate_bias_mean', 0.0)
        self.rate_bias_stddev = rospy.get_param('~rate_bias_stddev', 0.0)
        self.rate_rw_stddev = rospy.get_param('~rate_rw_stddev', 0.0)  # rad/s/sqrt(s)

        # --- Accel parameters ---
        self.accel_noise_stddev = rospy.get_param('~accel_noise_stddev', 0.0)
        self.accel_bias_mean = rospy.get_param('~accel_bias_mean', 0.0)
        self.accel_bias_stddev = rospy.get_param('~accel_bias_stddev', 0.0)
        self.accel_rw_stddev = rospy.get_param('~accel_rw_stddev', 0.0)  # m/s^2/sqrt(s)

        # Noise enabled flag
        self.enable_noise = True

        # Save original values for restore
        self.restore_values = {
            'rate_noise_stddev': self.rate_noise_stddev,
            'rate_bias_mean': self.rate_bias_mean,
            'rate_bias_stddev': self.rate_bias_stddev,
            'rate_rw_stddev': self.rate_rw_stddev,
            'accel_noise_stddev': self.accel_noise_stddev,
            'accel_bias_mean': self.accel_bias_mean,
            'accel_bias_stddev': self.accel_bias_stddev,
            'accel_rw_stddev': self.accel_rw_stddev
        }

        # Random-walk state variables
        self.rate_rw_bias = np.zeros(3)
        self.accel_rw_bias = np.zeros(3)

        # Timestamp for dt computation
        self.last_stamp = None
        self.default_dt = 1.0 / 200.0  # fallback dt

        # --- ROS publishers/subscribers ---
        self.enable_pub = rospy.Publisher('/enable_noise', Bool, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu1', Imu, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu0', Imu, self.imu_callback)

    # ---------- Utilities ----------
    def _sample_bias(self, bias_mean, bias_stddev):
        bias = gauss(bias_mean, bias_stddev)
        if uniform(0, 1) < 0.5:
            bias *= -1
        return bias

    def _get_dt(self, stamp):
        if self.last_stamp is None:
            self.last_stamp = stamp
            return self.default_dt
        dt = (stamp - self.last_stamp).to_sec()
        if dt <= 0.0 or dt > 1.0:
            dt = self.default_dt
        self.last_stamp = stamp
        return dt

    def restore_param_values(self):
        for k, v in self.restore_values.items():
            setattr(self, k, v)
        self.publish_param_values()

    def delault_param_values(self):
        for k in self.restore_values.keys():
            self.restore_values[k] = getattr(self, k)
        # Zero all live parameters
        self.rate_noise_stddev = 0.0
        self.rate_bias_mean = 0.0
        self.rate_bias_stddev = 0.0
        self.rate_rw_stddev = 0.0
        self.accel_noise_stddev = 0.0
        self.accel_bias_mean = 0.0
        self.accel_bias_stddev = 0.0
        self.accel_rw_stddev = 0.0
        self.rate_rw_bias = np.zeros(3)
        self.accel_rw_bias = np.zeros(3)
        self.publish_param_values()

    def publish_param_values(self):
        rospy.set_param('~rate_noise_stddev', self.rate_noise_stddev)
        rospy.set_param('~rate_bias_mean', self.rate_bias_mean)
        rospy.set_param('~rate_bias_stddev', self.rate_bias_stddev)
        rospy.set_param('~rate_rw_stddev', self.rate_rw_stddev)
        rospy.set_param('~accel_noise_stddev', self.accel_noise_stddev)
        rospy.set_param('~accel_bias_mean', self.accel_bias_mean)
        rospy.set_param('~accel_bias_stddev', self.accel_bias_stddev)
        rospy.set_param('~accel_rw_stddev', self.accel_rw_stddev)

    def enable_disable_noise(self):
        self.enable_noise = not self.enable_noise
        self.enable_pub.publish(Bool(data=self.enable_noise))

    def log_params(self):
        rospy.loginfo("------ Current IMU Noise Parameters ------")
        rospy.loginfo("rate_noise_stddev = %.6f" % self.rate_noise_stddev)
        rospy.loginfo("rate_bias_mean = %.6f" % self.rate_bias_mean)
        rospy.loginfo("rate_bias_stddev = %.6f" % self.rate_bias_stddev)
        rospy.loginfo("rate_rw_stddev = %.6f" % self.rate_rw_stddev)
        rospy.loginfo("accel_noise_stddev = %.6f" % self.accel_noise_stddev)
        rospy.loginfo("accel_bias_mean = %.6f" % self.accel_bias_mean)
        rospy.loginfo("accel_bias_stddev = %.6f" % self.accel_bias_stddev)
        rospy.loginfo("accel_rw_stddev = %.6f" % self.accel_rw_stddev)
        rospy.loginfo("----------------------------------------")

    # ---------- ROS callback ----------
    def imu_callback(self, imu_msg):
        if not self.enable_noise:
            self.imu_pub.publish(imu_msg)
            self._get_dt(imu_msg.header.stamp)  # update dt
            return

        imu_out = Imu()
        imu_out.header = imu_msg.header
        imu_out.orientation = imu_msg.orientation

        dt = self._get_dt(imu_msg.header.stamp)

        # Update random-walk biases
        self.rate_rw_bias += np.random.normal(0, self.rate_rw_stddev * np.sqrt(dt), 3)
        self.accel_rw_bias += np.random.normal(0, self.accel_rw_stddev * np.sqrt(dt), 3)

        # Angular velocity with noise + bias + RW
        imu_out.angular_velocity.x = imu_msg.angular_velocity.x + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev) + self.rate_rw_bias[0]
        imu_out.angular_velocity.y = imu_msg.angular_velocity.y + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev) + self.rate_rw_bias[1]
        imu_out.angular_velocity.z = imu_msg.angular_velocity.z + gauss(0, self.rate_noise_stddev) + self._sample_bias(self.rate_bias_mean, self.rate_bias_stddev) + self.rate_rw_bias[2]

        # Linear acceleration with noise + bias + RW
        imu_out.linear_acceleration.x = imu_msg.linear_acceleration.x + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev) + self.accel_rw_bias[0]
        imu_out.linear_acceleration.y = imu_msg.linear_acceleration.y + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev) + self.accel_rw_bias[1]
        imu_out.linear_acceleration.z = imu_msg.linear_acceleration.z + gauss(0, self.accel_noise_stddev) + self._sample_bias(self.accel_bias_mean, self.accel_bias_stddev) + self.accel_rw_bias[2]

        self.imu_pub.publish(imu_out)

    # ---------- Keyboard control ----------
    def restore_terminal_settings(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def handle_keyboard_input(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                    key = sys.stdin.read(1)
                    changed = False

                    # --- Gyro noise ---
                    if key == 'q': self.rate_noise_stddev += 0.0024; changed = True
                    elif key == 'a': self.rate_noise_stddev = max(0, self.rate_noise_stddev - 0.0024); changed = True
                    elif key == 'w': self.rate_bias_mean += 0.02; changed = True
                    elif key == 's': self.rate_bias_mean = max(0, self.rate_bias_mean - 0.02); changed = True
                    elif key == 'e': self.rate_bias_stddev += 0.02; changed = True
                    elif key == 'd': self.rate_bias_stddev = max(0, self.rate_bias_stddev - 0.02); changed = True
                    elif key == 'u': self.rate_rw_stddev += 0.0001; changed = True
                    elif key == 'j': self.rate_rw_stddev = max(0, self.rate_rw_stddev - 0.0001); changed = True

                    # --- Accel noise ---
                    elif key == 'r': self.accel_noise_stddev += 0.0283; changed = True
                    elif key == 'f': self.accel_noise_stddev = max(0, self.accel_noise_stddev - 0.0283); changed = True
                    elif key == 't': self.accel_bias_mean += 0.02; changed = True
                    elif key == 'g': self.accel_bias_mean = max(0, self.accel_bias_mean - 0.02); changed = True
                    elif key == 'y': self.accel_bias_stddev += 0.02; changed = True
                    elif key == 'h': self.accel_bias_stddev = max(0, self.accel_bias_stddev - 0.02); changed = True
                    elif key == 'i': self.accel_rw_stddev += 0.0001; changed = True
                    elif key == 'k': self.accel_rw_stddev = max(0, self.accel_rw_stddev - 0.0001); changed = True

                    # --- Enable/disable noise ---
                    elif key == ' ':
                        if not self.enable_noise:
                            self.enable_disable_noise()
                            self.restore_param_values()
                        else:
                            self.enable_disable_noise()
                            self.delault_param_values()
                        rospy.loginfo("Noise enabled: %s" % self.enable_noise)
                        changed = True

                    # --- Reset RW states ---
                    elif key == 'z':
                        self.rate_rw_bias[:] = 0
                        self.accel_rw_bias[:] = 0
                        rospy.loginfo("Random-walk states reset to zero")
                        changed = True

                    # --- Exit ---
                    elif key == '\x03':
                        break

                    # Log updated parameters
                    if changed:
                        self.log_params()
                        self.publish_param_values()

                rate.sleep()
        finally:
            self.restore_terminal_settings()

# ---------- Main ----------
if __name__ == '__main__':
    node = IMUNoiseParamControlNode()
    node.handle_keyboard_input()
