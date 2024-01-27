#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import HumanSkeleton, HumanSkeletonArray, Segment
from geometry_msgs.msg import Point
from std_msgs.msg import Header

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

class HandKeypointDetector(object):
    def __init__(self):

        self.bridge = CvBridge()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.image_sub = rospy.Subscriber('~input_image', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.keypoints_array_pub = rospy.Publisher('~output/hand_keypoints', HumanSkeletonArray, queue_size=1)
        self.vis_image_pub = rospy.Publisher('~output/debug_image', Image, queue_size=1)

    def image_callback(self, data):
        try:
            current_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        results = self.hands.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        keypoints_array_msg = HumanSkeletonArray()
        keypoints_array_msg.header = Header(stamp=rospy.Time.now(), frame_id=data.header.frame_id)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_msg = HumanSkeleton()
                hand_msg.header = Header(stamp=rospy.Time.now(), frame_id=data.header.frame_id)

                for id, lm in enumerate(hand_landmarks.landmark):
                    hand_msg.bone_names.append(f'landmark_{id}')
                    segment = Segment()
                    segment.start_point = Point(x=lm.x, y=lm.y, z=lm.z)
                    segment.end_point = Point(x=lm.x, y=lm.y, z=lm.z)
                    hand_msg.bones.append(segment)

                keypoints_array_msg.skeletons.append(hand_msg)
                self.mp_drawing.draw_landmarks(current_image, hand_landmarks, HAND_CONNECTIONS)

        # Always publish the keypoints array and visualization image
        self.keypoints_array_pub.publish(keypoints_array_msg)
        try:
            vis_image_msg = self.bridge.cv2_to_imgmsg(current_image, 'bgr8')
            self.vis_image_pub.publish(vis_image_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('hand_keypoint_detector')
    HandKeypointDetector()
    rospy.spin()
