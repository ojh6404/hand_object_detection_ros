#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters
import tf2_ros
import cv2
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import HumanSkeleton, HumanSkeletonArray, Segment
from geometry_msgs.msg import Point, Pose, PoseArray
from geometry_msgs.msg import TransformStamped

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
        self.with_tf = rospy.get_param('~with_tf', False)
        self.with_depth = True if self.with_tf else rospy.get_param('~with_depth', False)
        self.mp_drawing = mp.solutions.drawing_utils
        self.image_pub = rospy.Publisher('~output/debug_image', Image, queue_size=1)
        self.keypoints_array_pub = rospy.Publisher('~output/hand_keypoints', HumanSkeletonArray, queue_size=1)
        self.hand_center_pub = rospy.Publisher('~output/hand_center', PoseArray, queue_size=1)

        # tf listen and broadcast when with_tf is True
        if self.with_tf:
            assert self.with_depth, 'with_tf is True but with_depth is False'
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        image_sub = message_filters.Subscriber('~input_image', Image, buff_size=2**24)
        rect_sub = message_filters.Subscriber('~input_rects', RectArray, buff_size=2**24)
        self.subs = [image_sub, rect_sub]
        if self.with_depth:
            depth_sub = message_filters.Subscriber('~input_depth', Image, buff_size=2**24)
            cam_info_sub = message_filters.Subscriber('~input_cam_info', CameraInfo, buff_size=2**24)
            self.subs.append(depth_sub)
            self.subs.append(cam_info_sub)
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs, queue_size=1, slop=rospy.get_param('~slop', 0.1))
        self.ts.registerCallback(self.callback)

    def callback(self, *msgs):
        if self.with_depth:
            image_data, rect_data, depth_data, cam_info_data = msgs
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(cam_info_data)
        else:
            image_data, rect_data = msgs
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
            if self.with_depth:
                depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1") # TODO: passthrough
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Image dimensions
        img_height, img_width, _ = cv_image.shape
        keypoints_array_msg = HumanSkeletonArray(header=image_data.header)
        hand_center_msg = PoseArray(header=keypoints_array_msg.header)
        for rect in rect_data.rects:
            # Rescale ROI coordinates
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            roi = cv_image[y:y+h, x:x+w]

            # Process keypoints within ROI using MediaPipe
            results = self.hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # Draw keypoints
            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_msg = HumanSkeleton()
                    hand_msg.header = keypoints_array_msg.header
                    keypoints_array_msg.human_ids.append(hand_id)

                    # Scale keypoints back to the original image size
                    for lm in hand_landmarks.landmark:
                        lm.x = (lm.x * w + x) / img_width # scale to 0 ~ 1 with original image size
                        lm.y = (lm.y * h + y) / img_height

                    # Calculate palm center
                    hand_center_msg = PoseArray(header=keypoints_array_msg.header)
                    palm_cx, palm_cy = self.calc_palm_center(img_height, img_width, hand_landmarks)
                    # clip
                    palm_cx, palm_cy = np.clip(palm_cx, 0, img_width - 1), np.clip(palm_cy, 0, img_height - 1)
                    if self.with_depth:
                        palm_depth = depth_image[int(palm_cy), int(palm_cx)]
                        palm_xyz = camera_model.projectPixelTo3dRay((palm_cx, palm_cy))
                        palm_xyz = [i * palm_depth for i in palm_xyz]
                        hand_center_msg.poses.append(Pose(
                            position=Point(x=palm_xyz[0], y=palm_xyz[1], z=palm_xyz[2]),
                        ))
                        if self.with_tf:
                            # Publish tf, check if nan
                            if np.isnan(palm_xyz[0]) or np.isnan(palm_xyz[1]) or np.isnan(palm_xyz[2]):
                                rospy.logwarn('palm_xyz is nan, skip broadcasting tf')
                            else:
                                transform_stamped = TransformStamped()
                                transform_stamped.header.stamp = rospy.Time.now()
                                transform_stamped.header.frame_id = image_data.header.frame_id
                                transform_stamped.child_frame_id = f'hand_{hand_id}'
                                transform_stamped.transform.translation.x = palm_xyz[0]
                                transform_stamped.transform.translation.y = palm_xyz[1]
                                transform_stamped.transform.translation.z = palm_xyz[2]
                                transform_stamped.transform.rotation.x = 0
                                transform_stamped.transform.rotation.y = 0
                                transform_stamped.transform.rotation.z = 0
                                transform_stamped.transform.rotation.w = 1
                                self.tf_broadcaster.sendTransform(transform_stamped)
                    else:
                        hand_center_msg.poses.append(Pose(
                            position=Point(x=palm_cx, y=palm_cy, z=0),
                        ))

                    # Create bone segments
                    for start_id, end_id in CONNECTIONS:
                        lm_start = hand_landmarks.landmark[start_id]
                        lm_end = hand_landmarks.landmark[end_id]
                        lm_start_x = lm_start.x * img_width
                        lm_start_y = lm_start.y * img_height
                        lm_end_x = lm_end.x * img_width
                        lm_end_y = lm_end.y * img_height
                        if self.with_depth:
                            # 3D
                            # clip to image size
                            lm_start_depth = depth_image[np.clip(int(lm_start_y), 0, img_height - 1), np.clip(int(lm_start_x), 0, img_width - 1)]
                            lm_end_depth = depth_image[np.clip(int(lm_end_y), 0, img_height - 1), np.clip(int(lm_end_x), 0, img_width - 1)]
                            lm_start_xyz = camera_model.projectPixelTo3dRay((lm_start_x, lm_start_y))
                            lm_end_xyz = camera_model.projectPixelTo3dRay((lm_end_x, lm_end_y))
                            lm_start_xyz = [i * lm_start_depth for i in lm_start_xyz]
                            lm_end_xyz = [i * lm_end_depth for i in lm_end_xyz]
                            lm_start_x, lm_start_y = lm_start_xyz[0], lm_start_xyz[1]
                            lm_end_x, lm_end_y = lm_end_xyz[0], lm_end_xyz[1]
                            segment = Segment(
                                start_point=Point(x=lm_start_x, y=lm_start_y, z=lm_start_depth),
                                end_point=Point(x=lm_end_x, y=lm_end_y, z=lm_end_depth),
                            )
                        else:
                            # 2D
                            segment = Segment(
                                start_point=Point(x=lm_start_x, y=lm_start_y, z=0),
                                end_point=Point(x=lm_end_x, y=lm_end_y, z=0),
                            )
                        hand_msg.bone_names.append(f'{KEYPOINT_NAMES[start_id]}->{KEYPOINT_NAMES[end_id]}')
                        hand_msg.bones.append(segment)
                    keypoints_array_msg.skeletons.append(hand_msg)
                    # Draw keypoints
                    self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, HAND_CONNECTIONS)

        # Publish the detection image
        try:
            detection_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(detection_image_msg)
            self.keypoints_array_pub.publish(keypoints_array_msg)
            self.hand_center_pub.publish(hand_center_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

    def calc_palm_center(self, image_height, image_width, landmarks):
        palm_array = np.empty((0, 2), int)
        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            if index == 0:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:
                palm_array = np.append(palm_array, landmark_point, axis=0)
        # center of palm
        cx = int(np.mean(palm_array[:, 0]))
        cy = int(np.mean(palm_array[:, 1]))
        return cx, cy


if __name__ == '__main__':
    rospy.init_node('hand_keypoint_detection_node')
    node = HandKeypointDetectionNode()
    rospy.spin()
