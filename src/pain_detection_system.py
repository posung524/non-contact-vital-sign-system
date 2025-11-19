import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from datetime import datetime
import json
import math
import os
from kinect_data_receiver import KinectDataReceiver

class RightHandOnlyDetection:
    """只檢測右手的疼痛檢測系統"""
    
    def __init__(self):
        # 設置環境變數
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # MediaPipe初始化
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.touch_threshold = 50  # 觸碰判斷距離（像素）
        # 使用輕量級設定
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 檢測兩隻手但只使用右手
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        # Kinect數據接收器
        self.kinect_receiver = KinectDataReceiver()
        
        # 身體部位定義（英文顯示）
        self.body_parts_english = {
            'HEAD': 'Head',
            'NECK': 'Neck',
            'LEFT_SHOULDER': 'L.Shoulder',
            'RIGHT_SHOULDER': 'R.Shoulder',
            'LEFT_ARM': 'L.Arm',
            'RIGHT_ARM': 'R.Arm',
            'LEFT_ELBOW': 'L.Elbow',
            'RIGHT_ELBOW': 'R.Elbow',
            'LEFT_WRIST': 'L.Wrist',
            'RIGHT_WRIST': 'R.Wrist',
            'UPPER_CHEST': 'Upper Chest',
            'LOWER_CHEST': 'Lower Chest',
            'UPPER_ABDOMEN': 'Upper Abdomen',
            'LOWER_ABDOMEN': 'Lower Abdomen',
            'UPPER_BACK': 'Upper Back',
            'LOWER_BACK': 'Lower Back',
            'LEFT_HIP': 'L.Hip',
            'RIGHT_HIP': 'R.Hip',
            'LEFT_KNEE': 'L.Knee',
            'RIGHT_KNEE': 'R.Knee',
            'LEFT_ANKLE': 'L.Ankle',
            'RIGHT_ANKLE': 'R.Ankle',
            'LEFT_FOOT': 'L.Foot',
            'RIGHT_FOOT': 'R.Foot'
        }
        
        # 中文對照（用於記錄）
        self.body_parts_chinese = {
            'HEAD': '頭部', 'NECK': '頸部',
            'LEFT_SHOULDER': '左肩', 'RIGHT_SHOULDER': '右肩',
            'LEFT_ARM': '左臂', 'RIGHT_ARM': '右臂',
            'LEFT_ELBOW': '左肘', 'RIGHT_ELBOW': '右肘',
            'LEFT_WRIST': '左腕', 'RIGHT_WRIST': '右腕',
            'UPPER_CHEST': '上胸部', 'LOWER_CHEST': '下胸部',
            'UPPER_ABDOMEN': '上腹部', 'LOWER_ABDOMEN': '下腹部',
            'UPPER_BACK': '上背部', 'LOWER_BACK': '下背部',
            'LEFT_HIP': '左髖', 'RIGHT_HIP': '右髖',
            'LEFT_KNEE': '左膝', 'RIGHT_KNEE': '右膝',
            'LEFT_ANKLE': '左踝', 'RIGHT_ANKLE': '右踝',
            'LEFT_FOOT': '左腳', 'RIGHT_FOOT': '右腳'
        }
        
        # 檢測參數
        self.pain_records = []
        self.current_touch_start = None
        self.current_body_part = None
        self.pointing_threshold = 80
        self.hold_time_threshold = 2.5
        self.depth_tolerance = 120
        self.back_depth_threshold = 60
        
        # 系統狀態
        self.running = True
        self.frame_count = 0
        self.detection_count = 0
        self.right_hand_count = 0
        self.left_hand_ignored_count = 0
        
        print("Right Hand Only Detection System Initialized")
    
    def safe_int(self, value):
        """安全的整數轉換"""
        try:
            return int(round(float(value)))
        except (ValueError, TypeError):
            return 0
    
    def safe_point(self, pos):
        """安全的座標點轉換"""
        if pos is None:
            return None
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return (self.safe_int(pos[0]), self.safe_int(pos[1]))
        return None
    
    def get_landmark_position(self, landmarks, landmark_name):
        """獲取特定關鍵點的位置"""
        landmark_map = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        }
        
        if landmark_name in landmark_map:
            landmark = landmarks.landmark[landmark_map[landmark_name]]
            x = self.safe_int(landmark.x * 480)
            y = self.safe_int(landmark.y * 640)
            return (x, y)
        return None
    
    def is_right_hand(self, hand_landmarks, handedness_info):
        """判斷是否為右手"""
        # 使用handedness_info判斷
        if handedness_info and handedness_info.classification:
            hand_label = handedness_info.classification[0].label
            # MediaPipe的標籤是相對於攝像頭的，所以"Right"實際上是用戶的右手
            return hand_label == "Right"
        
        # 備用方法：如果handedness_info不可用，使用位置判斷
        # 獲取手腕位置
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        wrist_x = wrist.x * 480
        
        # 假設畫面中心為240，右手通常在畫面左側（用戶視角的右側）
        return wrist_x < 240
    
    def detect_right_hand_back_pointing(self, pose_landmarks):
        """檢測右手背部指向（當右手被遮擋時）"""
        if pose_landmarks is None:
            return None, None
        
        # 獲取右手腕和右肩位置
        right_shoulder = self.get_landmark_position(pose_landmarks, 'right_shoulder')
        right_wrist = self.get_landmark_position(pose_landmarks, 'right_wrist')
        
        if not all([right_shoulder, right_wrist]):
            return None, None
        
        # 檢查右手腕是否在身體後方
        wrist_depth = self.kinect_receiver.get_depth_at_point(right_wrist[0], right_wrist[1])
        shoulder_depth = self.kinect_receiver.get_depth_at_point(right_shoulder[0], right_shoulder[1])
        
        if not all([wrist_depth, shoulder_depth]):
            return None, None
        
        # 如果右手腕深度比右肩深度大，可能在指向背部
        if wrist_depth > shoulder_depth + self.back_depth_threshold:
            # 計算軀幹中心和區域
            left_shoulder = self.get_landmark_position(pose_landmarks, 'left_shoulder')
            left_hip = self.get_landmark_position(pose_landmarks, 'left_hip')
            right_hip = self.get_landmark_position(pose_landmarks, 'right_hip')
            
            if not all([left_shoulder, left_hip, right_hip]):
                return None, None
            
            # 計算軀幹中心
            center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
            hip_y = (left_hip[1] + right_hip[1]) // 2
            torso_height = hip_y - shoulder_y
            
            if torso_height <= 0:
                return None, None
            
            # 基於右手腕Y位置判斷背部區域
            wrist_y = right_wrist[1]
            relative_y = (wrist_y - shoulder_y) / torso_height
            
            if relative_y < 0.5:
                detected_part = 'UPPER_BACK'
                back_center_y = shoulder_y + int(torso_height * 0.25)
            else:
                detected_part = 'LOWER_BACK'
                back_center_y = shoulder_y + int(torso_height * 0.75)
            
            target_pos = (center_x, back_center_y)
            return detected_part, target_pos
        
        return None, None
    
    def calculate_torso_regions(self, pose_landmarks):
        """計算軀幹區域"""
        left_shoulder = self.get_landmark_position(pose_landmarks, 'left_shoulder')
        right_shoulder = self.get_landmark_position(pose_landmarks, 'right_shoulder')
        left_hip = self.get_landmark_position(pose_landmarks, 'left_hip')
        right_hip = self.get_landmark_position(pose_landmarks, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
        hip_y = (left_hip[1] + right_hip[1]) // 2
        torso_height = hip_y - shoulder_y
        
        if torso_height <= 0:
            return None
        
        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4
        
        regions = {
            'UPPER_CHEST': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.125)),
            'LOWER_CHEST': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.375)),
            'UPPER_ABDOMEN': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.625)),
            'LOWER_ABDOMEN': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.875)),
            'UPPER_BACK': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.25)),
            'LOWER_BACK': (self.safe_int(center_x), self.safe_int(shoulder_y + torso_height * 0.75))
        }
        
        return regions, shoulder_y, hip_y, torso_height
    
    def determine_front_or_back(self, finger_depth, pose_landmarks):
        """判斷指向正面還是背面"""
        left_shoulder = self.get_landmark_position(pose_landmarks, 'left_shoulder')
        right_shoulder = self.get_landmark_position(pose_landmarks, 'right_shoulder')
        
        if not all([left_shoulder, right_shoulder, finger_depth]):
            return None
        
        center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        center_y = (left_shoulder[1] + right_shoulder[1]) // 2
        center_depth = self.kinect_receiver.get_depth_at_point(center_x, center_y)
        
        if center_depth is None or center_depth <= 0:
            return None
        
        depth_diff = finger_depth - center_depth
        
        if depth_diff > self.back_depth_threshold:
            return 'back'
        elif abs(depth_diff) <= self.back_depth_threshold:
            return 'front'
        else:
            return None
    
    def get_right_hand_pointing_direction(self, hand_landmarks):
        """計算右手食指指向方向"""
        if hand_landmarks is None:
            return None, None
        
        finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        direction_x = finger_tip.x - finger_mcp.x
        direction_y = finger_tip.y - finger_mcp.y
        
        length = math.sqrt(direction_x**2 + direction_y**2)
        if length > 0:
            direction_x /= length
            direction_y /= length
        
        tip_x = self.safe_int(finger_tip.x * 480)
        tip_y = self.safe_int(finger_tip.y * 640)
        tip_pos = (tip_x, tip_y)
        
        return tip_pos, (direction_x, direction_y)
    
    def calculate_pointing_distance(self, finger_pos, finger_direction, target_pos):
        """計算指向距離"""
        if not all([finger_pos, finger_direction, target_pos]):
            return float('inf')
        
        to_target_x = target_pos[0] - finger_pos[0]
        to_target_y = target_pos[1] - finger_pos[1]
        
        dot_product = to_target_x * finger_direction[0] + to_target_y * finger_direction[1]
        
        if dot_product < 0:
            return float('inf')
        
        perpendicular_x = to_target_x - dot_product * finger_direction[0]
        perpendicular_y = to_target_y - dot_product * finger_direction[1]
        
        return math.sqrt(perpendicular_x**2 + perpendicular_y**2)
    
    # def detect_body_part_with_right_hand(self, finger_pos, finger_direction, finger_depth, pose_landmarks):
    #     """使用右手檢測指向的身體部位"""
    #     if not all([finger_pos, finger_direction, finger_depth, pose_landmarks]):
    #         return None, None
        
    #     # 判斷正面還是背面
    #     front_or_back = self.determine_front_or_back(finger_depth, pose_landmarks)
        
    #     # 計算軀幹區域
    #     torso_data = self.calculate_torso_regions(pose_landmarks)
        
    #     min_distance = float('inf')
    #     detected_part = None
    #     target_pos = None
        
    #     # 檢查軀幹區域
    #     if torso_data and front_or_back:
    #         regions, shoulder_y, hip_y, torso_height = torso_data
    #         finger_y = finger_pos[1]
            
    #         if shoulder_y <= finger_y <= hip_y:
    #             for region_name, region_center in regions.items():
    #                 if front_or_back == 'back' and 'BACK' not in region_name:
    #                     continue
    #                 if front_or_back == 'front' and 'BACK' in region_name:
    #                     continue
                    
    #                 distance = self.calculate_pointing_distance(finger_pos, finger_direction, region_center)
                    
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     detected_part = region_name
    #                     target_pos = region_center
        
    #     # 檢查其他身體部位
    #     if min_distance > self.pointing_threshold:
    #         other_parts = {
    #             'HEAD': self.get_landmark_position(pose_landmarks, 'nose'),
    #             'NECK': self.get_landmark_position(pose_landmarks, 'nose'),
    #             'LEFT_SHOULDER': self.get_landmark_position(pose_landmarks, 'left_shoulder'),
    #             'RIGHT_SHOULDER': self.get_landmark_position(pose_landmarks, 'right_shoulder'),
    #             'LEFT_ELBOW': self.get_landmark_position(pose_landmarks, 'left_elbow'),
    #             'RIGHT_ELBOW': self.get_landmark_position(pose_landmarks, 'right_elbow'),
    #             'LEFT_WRIST': self.get_landmark_position(pose_landmarks, 'left_wrist'),
    #             'RIGHT_WRIST': self.get_landmark_position(pose_landmarks, 'right_wrist'),
    #             'LEFT_HIP': self.get_landmark_position(pose_landmarks, 'left_hip'),
    #             'RIGHT_HIP': self.get_landmark_position(pose_landmarks, 'right_hip'),
    #             'LEFT_KNEE': self.get_landmark_position(pose_landmarks, 'left_knee'),
    #             'RIGHT_KNEE': self.get_landmark_position(pose_landmarks, 'right_knee'),
    #             'LEFT_ANKLE': self.get_landmark_position(pose_landmarks, 'left_ankle'),
    #             'RIGHT_ANKLE': self.get_landmark_position(pose_landmarks, 'right_ankle')
    #         }
            
    #         for part_name, part_pos in other_parts.items():
    #             if part_pos:
    #                 part_depth = self.kinect_receiver.get_depth_at_point(part_pos[0], part_pos[1])
    #                 if part_depth and abs(finger_depth - part_depth) < self.depth_tolerance:
    #                     distance = self.calculate_pointing_distance(finger_pos, finger_direction, part_pos)
    #                     if distance < min_distance:
    #                         min_distance = distance
    #                         detected_part = part_name
    #                         target_pos = part_pos
        
    #     if min_distance < self.pointing_threshold:
    #         return detected_part, target_pos
    #     return None, None
    def detect_body_part_by_touch(self, finger_pos, finger_depth, pose_landmarks):
        """透過碰觸方式偵測右手接觸的身體部位"""
        if not all([finger_pos, finger_depth, pose_landmarks]):
            return None, None

        min_distance = float('inf')
        detected_part = None
        target_pos = None

        # 檢查所有可能的部位（正面與背面）
        body_parts = {
            'HEAD': self.get_landmark_position(pose_landmarks, 'nose'),
            'NECK': self.get_landmark_position(pose_landmarks, 'nose'),
            'LEFT_SHOULDER': self.get_landmark_position(pose_landmarks, 'left_shoulder'),
            'RIGHT_SHOULDER': self.get_landmark_position(pose_landmarks, 'right_shoulder'),
            'LEFT_ELBOW': self.get_landmark_position(pose_landmarks, 'left_elbow'),
            'RIGHT_ELBOW': self.get_landmark_position(pose_landmarks, 'right_elbow'),
            'LEFT_WRIST': self.get_landmark_position(pose_landmarks, 'left_wrist'),
            'RIGHT_WRIST': self.get_landmark_position(pose_landmarks, 'right_wrist'),
            'LEFT_HIP': self.get_landmark_position(pose_landmarks, 'left_hip'),
            'RIGHT_HIP': self.get_landmark_position(pose_landmarks, 'right_hip'),
            'LEFT_KNEE': self.get_landmark_position(pose_landmarks, 'left_knee'),
            'RIGHT_KNEE': self.get_landmark_position(pose_landmarks, 'right_knee'),
            'LEFT_ANKLE': self.get_landmark_position(pose_landmarks, 'left_ankle'),
            'RIGHT_ANKLE': self.get_landmark_position(pose_landmarks, 'right_ankle'),
        }

        # 加上前胸腹部與背部區域中心點
        torso_data = self.calculate_torso_regions(pose_landmarks)
        if torso_data:
            torso_regions, *_ = torso_data
            body_parts.update(torso_regions)

        for part_name, part_pos in body_parts.items():
            if part_pos:
                dist = np.linalg.norm(np.array(finger_pos) - np.array(part_pos))
                if dist < self.touch_threshold and dist < min_distance:
                    min_distance = dist
                    detected_part = part_name
                    target_pos = part_pos

        if detected_part:
            return detected_part, target_pos
        else:
            return None, None
    
    def safe_draw_circle(self, image, center, radius, color, thickness):
        """安全的圓圈繪製"""
        center = self.safe_point(center)
        if center is None:
            return
        try:
            cv2.circle(image, center, radius, color, thickness)
        except Exception as e:
            print(f"Draw circle error: {e}")
    
    def safe_draw_text(self, image, text, position, font, scale, color, thickness):
        """安全的文字繪製"""
        position = self.safe_point(position)
        if position is None:
            return
        try:
            cv2.putText(image, str(text), position, font, scale, color, thickness)
        except Exception as e:
            print(f"Draw text error: {e}")
    
    def process_frame(self, rgb_frame, depth_frame):
        """處理單幀數據 - 只處理右手"""
        self.frame_count += 1
        
        if rgb_frame is None or depth_frame is None:
            return None
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        # MediaPipe處理
        try:
            pose_results = self.pose.process(rgb_image)
            hands_results = self.hands.process(rgb_image)
        except Exception as e:
            print(f"MediaPipe error: {e}")
            return None
        display_image = rgb_image
        # 檢測狀態
        pose_detected = pose_results.pose_landmarks is not None
        right_hand_detected = False
        
        if pose_detected:
            self.detection_count += 1
            try:
                self.mp_drawing.draw_landmarks(
                    display_image, 
                    pose_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
            except Exception as e:
                print(f"Pose drawing error: {e}")
        
        # 右手檢測 - 只處理右手
        right_finger_pos = None
        right_finger_direction = None
        right_finger_depth = None
        
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # 檢查是否為右手
                if self.is_right_hand(hand_landmarks, handedness):
                    right_hand_detected = True
                    self.right_hand_count += 1
                    
                    # 繪製右手（綠色）
                    try:
                        self.mp_drawing.draw_landmarks(
                            display_image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                    except Exception as e:
                        print(f"Right hand drawing error: {e}")
                    
                    # 獲取右手食指信息
                    right_finger_pos, right_finger_direction = self.get_right_hand_pointing_direction(hand_landmarks)
                    
                    if right_finger_pos:
                        right_finger_depth = self.kinect_receiver.get_depth_at_point(right_finger_pos[0], right_finger_pos[1])
                        
                        # 繪製右手指向線（黃色）
                        if right_finger_direction:
                            end_x = self.safe_int(right_finger_pos[0] + right_finger_direction[0] * 50)
                            end_y = self.safe_int(right_finger_pos[1] + right_finger_direction[1] * 50)
                            end_pos = (end_x, end_y)
                            
                            cv2.arrowedLine(display_image, right_finger_pos, end_pos, (0, 255, 255), 2)
                            self.safe_draw_circle(display_image, right_finger_pos, 4, (0, 255, 0), -1)
                            
                            # 顯示 "RIGHT HAND" 標籤
                            label_pos = (right_finger_pos[0] + 10, right_finger_pos[1] - 10)
                            self.safe_draw_text(display_image, "RIGHT", label_pos, 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    # 左手被忽略，但可以用灰色顯示
                    self.left_hand_ignored_count += 1
                    try:
                        self.mp_drawing.draw_landmarks(
                            display_image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(128, 128, 128), thickness=1)
                        )
                        
                        # 顯示 "LEFT (IGNORED)" 標籤
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        wrist_pos = (self.safe_int(wrist.x * 480), self.safe_int(wrist.y * 640))
                        label_pos = (wrist_pos[0] + 10, wrist_pos[1] - 10)
                        self.safe_draw_text(display_image, "LEFT (IGNORED)", label_pos, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                    except Exception as e:
                        print(f"Left hand drawing error: {e}")
        
        # 疼痛部位檢測 - 只使用右手
        detected_part = None
        target_pos = None
        
        if right_finger_pos and right_finger_direction and right_finger_depth and pose_detected:
            # 使用右手檢測
            # detected_part, target_pos = self.detect_body_part_with_right_hand(
            #     right_finger_pos, right_finger_direction, right_finger_depth, pose_results.pose_landmarks
            # )
            detected_part, target_pos = self.detect_body_part_by_touch(
                right_finger_pos, right_finger_depth, pose_results.pose_landmarks
            )
        elif not right_hand_detected and pose_detected:
            # 右手被遮擋時，嘗試背部檢測
            detected_part, target_pos = self.detect_right_hand_back_pointing(pose_results.pose_landmarks)
            if detected_part:
                self.safe_draw_text(display_image, "RIGHT HAND BACK MODE", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if detected_part:
            english_name = self.body_parts_english.get(detected_part, detected_part)
            current_time = time.time()
            
            if target_pos:
                self.safe_draw_circle(display_image, target_pos, 10, (0, 0, 255), 2)
                text_pos = (target_pos[0] - 30, target_pos[1] - 15)
                self.safe_draw_text(display_image, english_name, text_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if self.current_body_part != detected_part:
                self.current_body_part = detected_part
                self.current_touch_start = current_time
                self.safe_draw_text(display_image, f"RIGHT HAND -> {english_name}", (10, 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                hold_time = current_time - self.current_touch_start
                progress = min(hold_time / self.hold_time_threshold, 1.0)
                
                self.safe_draw_text(display_image, f"RIGHT HAND -> {english_name} ({hold_time:.1f}s)", (10, 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 進度條
                bar_x, bar_y = 10, 40
                bar_width, bar_height = 180, 12
                
                cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                fill_width = self.safe_int(bar_width * progress)
                color = (0, 255, 0) if progress < 1.0 else (0, 0, 255)
                cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
                
                if hold_time >= self.hold_time_threshold:
                    self.record_pain_location(detected_part)
                    self.current_touch_start = None
                    self.current_body_part = None
                    
                    self.safe_draw_text(display_image, f"RECORDED: {english_name}!", (10, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.current_touch_start = None
            self.current_body_part = None
        
        # 狀態顯示
        status_y = display_image.shape[0] - 120
        
        overlay = display_image.copy()
        cv2.rectangle(overlay, (5, status_y - 5), (400, display_image.shape[0] - 5), (0, 0, 0), -1)
        display_image = cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0)
        
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        right_hand_rate = (self.right_hand_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        self.safe_draw_text(display_image, f"Frame:{self.frame_count} Detection:{detection_rate:.0f}%", 
                           (10, status_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.safe_draw_text(display_image, f"Pose:{'Yes' if pose_detected else 'No'} RightHand:{'Yes' if right_hand_detected else 'No'}", 
                           (10, status_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.safe_draw_text(display_image, f"RightHand Count:{self.right_hand_count} LeftIgnored:{self.left_hand_ignored_count}", 
                           (10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.safe_draw_text(display_image, f"Records:{len(self.pain_records)} Current:{'Pointing' if self.current_body_part else 'None'}", 
                           (10, status_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        stats = self.kinect_receiver.get_stats()
        self.safe_draw_text(display_image, f"RGB:{stats['rgb_fps']}fps Depth:{stats['depth_fps']}fps", 
                           (10, status_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display_image
    
    def record_pain_location(self, body_part):
        """記錄疼痛部位"""
        chinese_name = self.body_parts_chinese.get(body_part, body_part)
        english_name = self.body_parts_english.get(body_part, body_part)
        
        pain_record = {
            'timestamp': datetime.now().isoformat(),
            'body_part': body_part,
            'chinese_name': chinese_name,
            'english_name': english_name,
            'detection_method': 'right_hand_only_detection'
        }
        self.pain_records.append(pain_record)
        
        print(f"RIGHT HAND Recorded: {english_name} ({chinese_name})")
        self.save_pain_records()
    
    def save_pain_records(self):
        """保存疼痛記錄"""
        try:
            with open('pain_records_right_hand.json', 'w', encoding='utf-8') as f:
                json.dump(self.pain_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Save failed: {e}")
    
    def load_pain_records(self):
        """載入疼痛記錄"""
        try:
            with open('pain_records_right_hand.json', 'r', encoding='utf-8') as f:
                self.pain_records = json.load(f)
            print(f"Loaded {len(self.pain_records)} right hand records")
        except FileNotFoundError:
            print("Creating new right hand record file")
        except Exception as e:
            print(f"Load failed: {e}")
    
    def run(self):
        """運行系統"""
        print("Starting Right Hand Only Detection System...")
        
        self.load_pain_records()
        
        print("Waiting for Kinect data...")
        while not self.kinect_receiver.is_data_available():
            time.sleep(0.1)
        
        print("Kinect ready, starting RIGHT HAND ONLY detection...")
        
        try:
            while self.running:
                rgb_frame = self.kinect_receiver.get_rgb_frame()
                depth_frame = self.kinect_receiver.get_depth_frame()
                
                if rgb_frame is not None and depth_frame is not None:
                    processed_frame = self.process_frame(rgb_frame, depth_frame)
                    
                    if processed_frame is not None:
                        cv2.imshow('RIGHT HAND ONLY Detection - 480x640', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.show_statistics()
                    elif key == ord('c'):
                        self.pain_records = []
                        self.save_pain_records()
                        print("RIGHT HAND records cleared")
                    elif key == ord('r'):
                        self.current_touch_start = None
                        self.current_body_part = None
                        print("RIGHT HAND status reset")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nUser exit")
        except Exception as e:
            print(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def show_statistics(self):
        """顯示統計"""
        print(f"\nRIGHT HAND ONLY Statistics:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Detection success: {self.detection_count} ({self.detection_count/self.frame_count*100:.1f}%)")
        print(f"  Right hand detected: {self.right_hand_count} ({self.right_hand_count/self.frame_count*100:.1f}%)")
        print(f"  Left hand ignored: {self.left_hand_ignored_count}")
        print(f"  Pain records: {len(self.pain_records)}")
        
        if self.pain_records:
            print("  Recent RIGHT HAND records:")
            for record in self.pain_records[-5:]:
                time_str = datetime.fromisoformat(record['timestamp']).strftime("%H:%M:%S")
                english_name = record.get('english_name', record['body_part'])
                chinese_name = record.get('chinese_name', record['body_part'])
                print(f"    {time_str} - RIGHT HAND -> {english_name} ({chinese_name})")
            
            # 統計各部位
            part_counts = {}
            for record in self.pain_records:
                part = record.get('english_name', record['body_part'])
                part_counts[part] = part_counts.get(part, 0) + 1
            
            print("\n  RIGHT HAND body part counts:")
            for part, count in sorted(part_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {part}: {count}")
    
    def cleanup(self):
        """清理資源"""
        print("Cleaning up RIGHT HAND ONLY system...")
        self.running = False
        cv2.destroyAllWindows()
        self.kinect_receiver.close()
        self.save_pain_records()
        print("RIGHT HAND ONLY system closed")

if __name__ == "__main__":
    print("RIGHT HAND ONLY Detection System")
    print("="*60)
    print("Key Features:")
    print("  • ONLY detects RIGHT HAND pointing")
    print("  • LEFT HAND is ignored (shown in gray)")
    print("  • Clear visual distinction:")
    print("    - RIGHT HAND: Green landmarks + Yellow arrow")
    print("    - LEFT HAND: Gray landmarks + 'IGNORED' label")
    print("  • Enhanced back detection for right hand")
    print("  • Separate statistics for right hand only")
    print()
    print("Visual Indicators:")
    print("  • GREEN: Right hand landmarks and connections")
    print("  • YELLOW: Right hand pointing arrow")
    print("  • GRAY: Left hand (ignored)")
    print("  • RED: Target body part circles")
    print("  • 'RIGHT HAND ->' prefix in pointing text")
    print()
    print("Right Hand Detection Modes:")
    print("  1. Normal: Right finger pointing (front detection)")
    print("  2. Back: Right wrist behind body (back detection)")
    print("  3. Auto-switching when hand visibility changes")
    print()
    print("Usage:")
    print("  1. Stand 1.5-3m from Kinect")
    print("  2. Use ONLY your RIGHT HAND to point")
    print("  3. LEFT HAND will be visually ignored")
    print("  4. Hold position for 2.5 seconds to record")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Show RIGHT HAND statistics")
    print("  'c' - Clear RIGHT HAND records")
    print("  'r' - Reset RIGHT HAND status")
    print("="*60)
    
    try:
        system = RightHandOnlyDetection()
        system.run()
    except Exception as e:
        print(f"RIGHT HAND ONLY system failed: {e}")
        import traceback
        traceback.print_exc()