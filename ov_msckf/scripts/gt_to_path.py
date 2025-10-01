#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path

# class TransformToPath:
#     def __init__(self):
#         rospy.init_node("transform_to_path")
#         self.path_pub = rospy.Publisher("/vicon_path", Path, queue_size=10)
#         self.path = Path()
#         self.path.header.frame_id = "imu"  # or "map", depending on your tf tree
#         rospy.Subscriber("/vicon/firefly_sbx/firefly_sbx", TransformStamped, self.callback)
#         self.initial_offset = None

#     def callback(self, msg):
#         if self.initial_offset is None:
#             self.initial_offset = msg.transform.translation

#         pose = PoseStamped()
#         pose.header.stamp = rospy.Time.now()
#         pose.header.frame_id = msg.header.frame_id

#         pose.pose.position.x = msg.transform.translation.x - self.initial_offset.x
#         pose.pose.position.y = msg.transform.translation.y - self.initial_offset.y
#         pose.pose.position.z = msg.transform.translation.z - self.initial_offset.z

#         pose.pose.orientation = msg.transform.rotation

#         self.path.header.stamp = rospy.Time.now()
#         self.path.poses.append(pose)
#         self.path_pub.publish(self.path)


# if __name__ == "__main__":
#     TransformToPath()
#     rospy.spin()

class TransformToPath:
    def __init__(self):
        rospy.init_node("transform_to_path")
        self.path_pub = rospy.Publisher("/vicon_path", Path, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "global"

        self.initial_offset = None  # Save first GT position only once
        rospy.Subscriber("/vicon/firefly_sbx/firefly_sbx", TransformStamped, self.callback)

    def callback(self, msg):
        # Save only the very first position as origin
        if self.initial_offset is None:
            self.initial_offset = msg.transform.translation
            rospy.loginfo("Initial GT position saved as origin: x=%.3f y=%.3f z=%.3f",
                          self.initial_offset.x, self.initial_offset.y, self.initial_offset.z)

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"

        # Subtract *fixed* initial offset
        pose.pose.position.x = msg.transform.translation.x - self.initial_offset.x
        pose.pose.position.y = msg.transform.translation.y - self.initial_offset.y
        pose.pose.position.z = msg.transform.translation.z - self.initial_offset.z

        pose.pose.orientation = msg.transform.rotation

        self.path.header.stamp = rospy.Time.now()
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)

if __name__ == "__main__":
    TransformToPath()
    rospy.spin()
