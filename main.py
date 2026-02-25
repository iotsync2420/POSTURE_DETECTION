import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import numpy as np
import time
from collections import deque

class PostureAnalyzer:
    def __init__(self):
        # Posture definitions with health impacts (DEFINE FIRST!)
        self.posture_impacts = {
            'slouching_laptop': 'Back and Neck Pain',
            'forward_head': 'Cervical spine strain and headaches',
            'rounded_shoulders': 'Poor breathing posture',
            'flat_back': 'Chronic lower back pain',
            'poking_chin': 'Neck strain and tension headaches',
            'bent_shoulders_sitting': 'Upper back pain and fatigue',
            'bent_shoulders_standing': 'Muscle imbalance and pain'
        }
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Posture tracking with smoothing
        self.posture_history = deque(maxlen=10)
        self.current_posture = None
        self.posture_start_time = None
        self.current_issues = []
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points - optimized"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180.0 else 360 - angle
    
    def get_distance(self, p1, p2):
        """Calculate Euclidean distance"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_body_position(self, landmarks):
        """Detect if person is sitting or standing"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Calculate knee angle
        knee_angle = self.calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )
        
        # If knee angle < 120, likely sitting
        return 'sitting' if knee_angle < 120 else 'standing'
    
    def analyze_comprehensive_posture(self, landmarks):
        """Enhanced posture analysis with multiple conditions"""
        issues = []
        
        # Extract key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # Detect sitting or standing
        position = self.detect_body_position(landmarks)
        
        # Average ear position
        ear_x = (left_ear.x + right_ear.x) / 2
        ear_y = (left_ear.y + right_ear.y) / 2
        
        # Average shoulder position
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Average hip position
        hip_x = (left_hip.x + right_hip.x) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        # 1. FORWARD HEAD POSTURE
        ear_shoulder_horizontal = abs(ear_x - shoulder_x)
        if ear_shoulder_horizontal > 0.08:
            issues.append(('forward_head', 
                          f"Forward Head Posture ({position})"))
        
        # 2. POKING CHIN (nose significantly forward of ears)
        nose_ear_distance = abs(nose.x - ear_x)
        if nose_ear_distance > 0.12:
            issues.append(('poking_chin', 
                          f"Poking Chin Detected ({position})"))
        
        # 3. ROUNDED SHOULDERS
        shoulder_angle = self.calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [shoulder_x, shoulder_y],
            [right_shoulder.x, right_shoulder.y]
        )
        
        # Check shoulder forward roll
        shoulder_forward = (left_shoulder.z + right_shoulder.z) / 2
        
        if shoulder_angle < 160 or shoulder_forward > 0.1:
            if position == 'sitting':
                issues.append(('bent_shoulders_sitting', 
                              "Rounded Shoulders (Sitting)"))
            else:
                issues.append(('bent_shoulders_standing', 
                              "Rounded Shoulders (Standing)"))
        
        # 4. SLOUCHING OVER LAPTOP (forward lean while sitting)
        if position == 'sitting':
            upper_body_angle = self.calculate_angle(
                [shoulder_x, shoulder_y],
                [hip_x, hip_y],
                [hip_x, hip_y + 0.1]
            )
            
            # Check if leaning forward significantly
            shoulder_hip_horizontal = abs(shoulder_x - hip_x)
            if upper_body_angle < 155 or shoulder_hip_horizontal > 0.15:
                issues.append(('slouching_laptop', 
                              "Slouching Over Laptop"))
        
        # 5. FLAT BACK POSTURE (excessive straightening, loss of natural curve)
        spine_angle = self.calculate_angle(
            [shoulder_x, shoulder_y],
            [hip_x, hip_y],
            [hip_x, hip_y + 0.1]
        )
        
        # Flat back: too straight (> 175 degrees) with pelvis tucked
        if spine_angle > 175:
            pelvis_tuck = (hip_y - shoulder_y) / abs(hip_x - shoulder_x + 0.001)
            if pelvis_tuck > 1.5:
                issues.append(('flat_back', 
                              "Flat Back Posture"))
        
        # 6. GENERAL SLOUCHING (if standing)
        if position == 'standing':
            if spine_angle < 160:
                issues.append(('slouching_laptop', 
                              "Slouching Posture (Standing)"))
        
        return issues if issues else [('good_posture', "Good Posture! Keep it up!")]
    
    def smooth_detection(self, current_issues):
        """Smooth posture detection over time"""
        if not current_issues:
            return None
        
        # Add to history
        primary_issue = current_issues[0][0]
        self.posture_history.append(primary_issue)
        
        # Get most common posture in recent history
        if len(self.posture_history) >= 5:
            from collections import Counter
            most_common = Counter(self.posture_history).most_common(1)[0][0]
            return most_common
        
        return primary_issue
    
    def draw_enhanced_ui(self, frame, issues, fps, position):
        """Enhanced UI with color-coded feedback"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Status bar at top
        cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # FPS and Position
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Position: {position.upper()}", (w - 250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Main feedback
        y_offset = 30
        for issue_type, issue_text in issues:
            if issue_type == 'good_posture':
                color = (0, 255, 0)
                cv2.putText(frame, issue_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            else:
                color = (0, 0, 255)
                # Issue title
                cv2.putText(frame, f"ALERT: {issue_text}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Health impact
                if issue_type in self.posture_impacts:
                    impact = self.posture_impacts[issue_type]
                    cv2.putText(frame, f"Impact: {impact}", (20, y_offset + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    y_offset += 60
                else:
                    y_offset += 35
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit", (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Optimized main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prev_time = 0
        frame_skip = 2  # Process every 2nd frame for optimization
        frame_count = 0
        
        print("=" * 60)
        print("POSTURE CORRECTION SYSTEM - PHYSIOTHERAPY EDITION")
        print("=" * 60)
        print("\nDetecting:")
        print("  • Forward Head Posture")
        print("  • Poking Chin")
        print("  • Rounded Shoulders (Sitting/Standing)")
        print("  • Slouching Over Laptop")
        print("  • Flat Back Posture")
        print("\nPress 'Q' to quit\n")
        
        # Initialize issues
        issues = [('no_detection', "Initializing...")]
        position = 'unknown'
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Process every nth frame
            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 100, 255), thickness=2)
                    )
                    
                    # Analyze posture
                    issues = self.analyze_comprehensive_posture(results.pose_landmarks.landmark)
                    position = self.detect_body_position(results.pose_landmarks.landmark)
                    
                    # Smooth detection
                    self.smooth_detection(issues)
                    self.current_issues = issues
                else:
                    issues = [('no_detection', "No person detected - Stand in frame")]
                    position = 'unknown'
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Draw UI
            frame = self.draw_enhanced_ui(frame, issues, fps, position)
            
            cv2.imshow('Posture Correction - Physiotherapy System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended. Take care of your posture!")

if __name__ == "__main__":
    try:
        analyzer = PostureAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()

