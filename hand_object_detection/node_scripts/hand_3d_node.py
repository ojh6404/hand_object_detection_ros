#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters as mf
import tf
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from image_geometry import PinholeCameraModel
from hand_object_detection_ros.msg import HandDetectionArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from scipy.signal import savgol_filter


class Hand3DNode(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.scale = rospy.get_param("~scale", 0.001)
        self.slop = rospy.get_param("~slop", 0.15)
        self.apply_filter = rospy.get_param("~apply_filter", False)
        if self.apply_filter:
            self.window_size = rospy.get_param("~window_size", 5)
            self.poly_order = rospy.get_param("~poly_order", 3)
            self.data_queue = []

        # Subscribe to the camera info and depth image topics
        self.info_sub = mf.Subscriber("~camera_info", CameraInfo, buff_size=2**24)
        self.depth_sub = mf.Subscriber("~input_depth", Image, buff_size=2**24)
        self.hand_sub = mf.Subscriber("~input_detections", HandDetectionArray, buff_size=2**24)
        self.ts = mf.ApproximateTimeSynchronizer(
            [self.info_sub, self.depth_sub, self.hand_sub], queue_size=1, slop=self.slop
        )
        self.ts.registerCallback(self.callback)

        # Publisher for Pose
        self.pose_array_pub = rospy.Publisher("~hand_pose", PoseArray, queue_size=10)

    def apply_savgol_filter(self, data):
        if len(data) < self.window_size:
            return data[-1]
        else:
            filtered_data = savgol_filter(np.array(data), self.window_size, self.poly_order)
            return filtered_data[-1]


    def callback(self, cam_info_data, depth_data, hand_data):
        # Extract the camera frame from the image message
        camera_frame = depth_data.header.frame_id
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(cam_info_data)

        try:
            # Convert the depth image to a Numpy array
            cv_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(e)

        for detection in hand_data.detections:
            # Get the depth value at the 2D point
            point_2d = (detection.pose.position.x, detection.pose.position.y)
            # clip the point to the image size
            point_2d = (
                min(max(detection.pose.position.x, 0), cv_image.shape[1] - 1),
                min(max(detection.pose.position.y, 0), cv_image.shape[0] - 1),
            )
            depth = cv_image[int(point_2d[1]), int(point_2d[0])]
            if np.isnan(depth) or (depth == 0.0):
                continue

            # Calculate 3D coordinates in the camera frame
            x_cam = (point_2d[0] - camera_model.cx()) * depth / camera_model.fx()
            y_cam = (point_2d[1] - camera_model.cy()) * depth / camera_model.fy()
            z_cam = depth

            if self.apply_filter:
                self.data_queue.append([x_cam, y_cam, z_cam])
                if len(self.data_queue) > self.window_size:
                    self.data_queue.pop(0)
                x_cam = self.apply_savgol_filter([x[0] for x in self.data_queue])
                y_cam = self.apply_savgol_filter([x[1] for x in self.data_queue])
                z_cam = self.apply_savgol_filter([x[2] for x in self.data_queue])

            try:
                # Create PoseArray message and publish it
                pose_msg = Pose()
                pose_msg.position.x = x_cam * self.scale
                pose_msg.position.y = y_cam * self.scale
                pose_msg.position.z = z_cam * self.scale
                pose_msg.orientation = detection.pose.orientation

                pose_array_msg = PoseArray()
                pose_array_msg.header.frame_id = camera_frame
                pose_array_msg.header.stamp = rospy.Time.now()
                pose_array_msg.poses.append(pose_msg)

                self.pose_array_pub.publish(pose_array_msg)

                # Broadcast the transform
                self.tf_broadcaster.sendTransform(
                    (pose_msg.position.x, pose_msg.position.y, pose_msg.position.z),
                    (pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w),
                    rospy.Time.now(),
                    detection.hand,
                    camera_frame,
                )

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)


if __name__ == "__main__":
    rospy.init_node("hand_3d_node")
    node = Hand3DNode()
    rospy.loginfo("Hand 3D node started")
    rospy.spin()
