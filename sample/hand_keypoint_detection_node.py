#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters
import cv2
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import HumanSkeleton, HumanSkeletonArray, Segment
from geometry_msgs.msg import Point
from std_msgs.msg import Header

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# mediapipe hands landmarks, id to name
# https://google.github.io/mediapipe/solutions/hands.html
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_finger_mcp",
    "pinky_finger_pip",
    "pinky_finger_dip",
    "pinky_finger_tip",
]

CONNECTIONS = [
    (0, 1), # wrist -> thumb_cmc
    (1, 2), # thumb_cmc -> thumb_mcp
    (2, 3), # thumb_mcp -> thumb_ip
    (3, 4), # thumb_ip -> thumb_tip
    (0, 5), # wrist -> index_finger_mcp
    (5, 6), # index_finger_mcp -> index_finger_pip
    (6, 7), # index_finger_pip -> index_finger_dip
    (7, 8), # index_finger_dip -> index_finger_tip
    (0, 9), # wrist -> middle_finger_mcp
    (9, 10), # middle_finger_mcp -> middle_finger_pip
    (10, 11), # middle_finger_pip -> middle_finger_dip
    (11, 12), # middle_finger_dip -> middle_finger_tip
    (0, 13), # wrist -> ring_finger_mcp
    (13, 14), # ring_finger_mcp -> ring_finger_pip
    (14, 15), # ring_finger_pip -> ring_finger_dip
    (15, 16), # ring_finger_dip -> ring_finger_tip
    (0, 17), # wrist -> pinky_finger_mcp
    (17, 18), # pinky_finger_mcp -> pinky_finger_pip
    (18, 19), # pinky_finger_pip -> pinky_finger_dip
    (19, 20), # pinky_finger_dip -> pinky_finger_tip
]

class HandKeypointDetectionNode(object):
    def __init__(self):

        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=rospy.get_param('~min_detection_confidence', 0.5),
            min_tracking_confidence=rospy.get_param('~min_tracking_confidence', 0.5),
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.image_pub = rospy.Publisher('~output/debug_image', Image, queue_size=1)
        self.keypoints_array_pub = rospy.Publisher('~output/hand_keypoints', HumanSkeletonArray, queue_size=1)

        image_sub = message_filters.Subscriber('~input_image', Image, buff_size=2**24)
        rect_sub = message_filters.Subscriber('~input_rects', RectArray, buff_size=2**24)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, rect_sub], queue_size=1, slop=rospy.get_param('~slop', 0.1))
        self.ts.registerCallback(self.callback)

    def callback(self, image_data, rect_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Image dimensions
        img_height, img_width, _ = cv_image.shape
        keypoints_array_msg = HumanSkeletonArray()
        keypoints_array_msg.header = Header(stamp=rospy.Time.now(), frame_id=image_data.header.frame_id)
        for rect in rect_data.rects:
            # Rescale ROI coordinates
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            roi = cv_image[y:y+h, x:x+w]

            # Process keypoints within ROI using MediaPipe
            results = self.hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # Draw keypoints
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_msg = HumanSkeleton()
                    hand_msg.header = Header(stamp=rospy.Time.now(), frame_id=image_data.header.frame_id)
                    # Scale keypoints back to the original image size
                    for lm in hand_landmarks.landmark:
                        lm.x = (lm.x * w + x) / img_width
                        lm.y = (lm.y * h + y) / img_height
                    # Draw keypoints
                    for start_id, end_id in CONNECTIONS:
                        lm_start = hand_landmarks.landmark[start_id]
                        lm_end = hand_landmarks.landmark[end_id]
                        # to real pixel coordinates
                        segment = Segment(
                            start_point=Point(x=lm_start.x * img_width, y=lm_start.y * img_height, z=lm_start.z),
                            end_point=Point(x=lm_end.x * img_width, y=lm_end.y * img_height, z=lm_end.z)
                        )
                        hand_msg.bone_names.append(f'{KEYPOINT_NAMES[start_id]}->{KEYPOINT_NAMES[end_id]}')
                        hand_msg.bones.append(segment)
                    keypoints_array_msg.skeletons.append(hand_msg)
                    self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, HAND_CONNECTIONS)

        # Publish the detection image
        try:
            detection_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(detection_image_msg)
            self.keypoints_array_pub.publish(keypoints_array_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('hand_keypoint_detection_node')
    node = HandKeypointDetectionNode()
    rospy.spin()
