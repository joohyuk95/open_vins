#!/usr/bin/env python

import rosbag
import rospy

# Input bag files
rgbdbag_path = 'rgbd.bag'
imubag_path = 'imu_data.bag'

# Output bag file
newbag_path = 'new.bag'

# Open the input bags
rgbdbag = rosbag.Bag(rgbdbag_path)
imubag = rosbag.Bag(imubag_path)

# Get the start time of the rgbd.bag
start_time = rgbdbag.get_start_time()

# Create a new bag for writing
newbag = rosbag.Bag(newbag_path, 'w')

# Iterate over messages in rgbd.bag and write to the new bag
for topic, msg, t in rgbdbag.read_messages():
    # Adjust the timestamp to be synchronized with imu_data.bag
    t -= rospy.Duration.from_sec(start_time - imubag.get_start_time())
    newbag.write(topic, msg, t)

# Iterate over messages in imu_data.bag and write to the new bag
for topic, msg, t in imubag.read_messages():
    newbag.write(topic, msg, t)

# Close the bags
rgbdbag.close()
imubag.close()
newbag.close()

print("New bag file created:", newbag_path)
