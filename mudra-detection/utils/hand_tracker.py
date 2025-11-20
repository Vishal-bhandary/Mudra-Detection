"""
Enhanced hand tracking utility - kept for backward compatibility
This is now integrated into the main detection script
"""
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    """Legacy hand tracker - use EnhancedHandTracker for better results"""
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        
        print("⚠️  Using legacy HandTracker. Consider using EnhancedHandTracker for better accuracy.")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_hands(self, image, draw=True):
        """Detect hands in image and optionally draw landmarks"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image, results
    
    def get_landmarks(self, results):
        """Extract landmark coordinates from results"""
        if not results.multi_hand_landmarks:
            return []
        
        landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(np.array(landmarks))
        
        return landmarks_list
    
    def get_flattened_landmarks(self, results, fixed_size=126):
        """Get flattened landmarks (legacy method)"""
        landmarks_flat = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks_flat.extend([landmark.x, landmark.y, landmark.z])
        
        if len(landmarks_flat) < fixed_size:
            landmarks_flat.extend([0] * (fixed_size - len(landmarks_flat)))
        elif len(landmarks_flat) > fixed_size:
            landmarks_flat = landmarks_flat[:fixed_size]
        
        return np.array(landmarks_flat)
    
    def get_hand_info(self, results):
        """Get hand information (handedness)"""
        if not results.multi_handedness:
            return []
        
        hand_info = []
        for idx, hand_handedness in enumerate(results.multi_handedness):
            handedness = hand_handedness.classification[0]
            hand_info.append({
                'index': idx,
                'label': handedness.label,
                'score': handedness.score
            })
        
        return hand_info
    
    def draw_bounding_box(self, image, results):
        """Draw bounding boxes around detected hands"""
        if not results.multi_hand_landmarks:
            return image
        
        h, w, c = image.shape
        
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return image
    
    def close(self):
        """Release resources"""
        self.hands.close()

# Enhanced version with feature engineering
class EnhancedHandTracker(HandTracker):
    """
    Enhanced hand tracker with comprehensive feature extraction
    Includes normalized landmarks, distances, and angles
    """
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Higher complexity for better accuracy
        )
    
    def calculate_distances(self, landmarks):
        """Calculate distances between key points"""
        distances = []
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]
        
        # Wrist to fingertips
        for tip in fingertips:
            dist = np.sqrt(
                (landmarks[tip][0] - wrist[0])**2 + 
                (landmarks[tip][1] - wrist[1])**2
            )
            distances.append(dist)
        
        # Between adjacent fingertips
        for i in range(len(fingertips)-1):
            dist = np.sqrt(
                (landmarks[fingertips[i]][0] - landmarks[fingertips[i+1]][0])**2 + 
                (landmarks[fingertips[i]][1] - landmarks[fingertips[i+1]][1])**2
            )
            distances.append(dist)
        
        return distances
    
    def calculate_angles(self, landmarks):
        """Calculate angles between finger segments"""
        angles = []
        fingers = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        for finger in fingers:
            for i in range(len(finger) - 2):
                p1 = np.array(landmarks[finger[i]])[:2]
                p2 = np.array(landmarks[finger[i+1]])[:2]
                p3 = np.array(landmarks[finger[i+2]])[:2]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        return angles
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist and hand size"""
        landmarks = np.array(landmarks)
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        distances = np.sqrt(np.sum(landmarks**2, axis=1))
        max_dist = np.max(distances) + 1e-6
        landmarks = landmarks / max_dist
        
        return landmarks.flatten()
    
    def extract_enhanced_features(self, results):
        """
        Extract comprehensive features matching training format
        
        Returns:
            features: numpy array of shape (176,)
            num_hands: int
        """
        if not results.multi_hand_landmarks:
            return None, 0
        
        all_features = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks)
            
            # Normalize landmarks (63 features)
            norm_landmarks = self.normalize_landmarks(landmarks)
            
            # Calculate distances (10 features)
            distances = self.calculate_distances(landmarks)
            
            # Calculate angles (15 features)
            angles = self.calculate_angles(landmarks)
            
            # Combine: 63 + 10 + 15 = 88 features per hand
            features = list(norm_landmarks) + distances + angles
            all_features.extend(features)
        
        # Pad to expected size (2 hands worth = 176 features)
        expected_size = 2 * (63 + 10 + 15)
        
        if len(all_features) < expected_size:
            all_features.extend([0] * (expected_size - len(all_features)))
        elif len(all_features) > expected_size:
            all_features = all_features[:expected_size]
        
        return np.array(all_features, dtype=np.float32), len(results.multi_hand_landmarks)