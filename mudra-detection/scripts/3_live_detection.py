"""
Enhanced Real-time Bharatanatyam Mudra Detection System
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import os
import sys
from collections import deque, Counter
import mediapipe as mp

class EnhancedHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
    def calculate_distances(self, landmarks):
        """Calculate distances between key points"""
        distances = []
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]
        
        for tip in fingertips:
            dist = np.sqrt(
                (landmarks[tip][0] - wrist[0])**2 + 
                (landmarks[tip][1] - wrist[1])**2
            )
            distances.append(dist)
        
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
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
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
        """Normalize landmarks relative to wrist"""
        landmarks = np.array(landmarks)
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        distances = np.sqrt(np.sum(landmarks**2, axis=1))
        max_dist = np.max(distances) + 1e-6
        landmarks = landmarks / max_dist
        
        return landmarks.flatten()
    
    def find_hands(self, image, draw=True):
        """Detect hands and return results"""
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
    
    def extract_enhanced_features(self, results):
        """Extract comprehensive features matching training format"""
        if not results.multi_hand_landmarks:
            return None, 0
        
        all_features = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks)
            
            # Normalize landmarks
            norm_landmarks = self.normalize_landmarks(landmarks)
            
            # Calculate distances
            distances = self.calculate_distances(landmarks)
            
            # Calculate angles
            angles = self.calculate_angles(landmarks)
            
            # Combine features
            features = list(norm_landmarks) + distances + angles
            all_features.extend(features)
        
        # Pad to expected size (2 hands worth)
        expected_size = 2 * (63 + 10 + 15)  # 176
        if len(all_features) < expected_size:
            all_features.extend([0] * (expected_size - len(all_features)))
        elif len(all_features) > expected_size:
            all_features = all_features[:expected_size]
        
        return np.array(all_features, dtype=np.float32), len(results.multi_hand_landmarks)
    
    def draw_bounding_box(self, image, results):
        """Draw bounding boxes around hands"""
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
        self.hands.close()

class EnhancedMudraDetector:
    def __init__(self, model_dir='models'):
        """Initialize with separate models for single/double hands"""
        
        print("Loading models and components...")
        
        self.models = {}
        self.label_encoders = {}
        self.model_info = {}
        
        # Load both single and double hand models
        for model_type in ['single', 'double']:
            try:
                model_path = os.path.join(model_dir, f'mudra_model_{model_type}.h5')
                encoder_path = os.path.join(model_dir, f'label_encoder_{model_type}.pkl')
                info_path = os.path.join(model_dir, f'model_info_{model_type}.json')
                
                if not os.path.exists(model_path):
                    print(f"⚠️  {model_type.title()} hand model not found, skipping...")
                    continue
                
                self.models[model_type] = keras.models.load_model(model_path)
                
                with open(encoder_path, 'rb') as f:
                    self.label_encoders[model_type] = pickle.load(f)
                
                with open(info_path, 'r') as f:
                    self.model_info[model_type] = json.load(f)
                
                print(f"✓ {model_type.title()} hand model loaded ({len(self.label_encoders[model_type].classes_)} classes)")
                
            except Exception as e:
                print(f"Error loading {model_type} hand model: {e}")
        
        if not self.models:
            raise ValueError("No models loaded! Please train models first.")
        
        # Initialize hand tracker
        self.hand_tracker = EnhancedHandTracker()
        print("✓ Enhanced hand tracker initialized")
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=7)
        self.confidence_threshold = 0.5
        
    def predict(self, features, num_hands):
        """Predict mudra using appropriate model"""
        
        model_type = 'single' if num_hands == 1 else 'double'
        
        if model_type not in self.models:
            return "Model not available", 0.0, {}
        
        # Reshape for model
        features_input = features.reshape(1, -1)
        
        # Predict
        predictions = self.models[model_type].predict(features_input, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        mudra_name = self.label_encoders[model_type].inverse_transform([predicted_idx])[0]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions)[-5:][::-1]
        top_predictions = {
            self.label_encoders[model_type].inverse_transform([i])[0]: float(predictions[i])
            for i in top_5_idx
        }
        
        return mudra_name, confidence, top_predictions
    
    def smooth_prediction(self, mudra_name, confidence, num_hands):
        """Smooth predictions with voting and temporal filtering"""
        self.prediction_history.append((mudra_name, confidence, num_hands))
        
        if len(self.prediction_history) < 4:
            return mudra_name, confidence
        
        # Filter by number of hands (only consider recent predictions with same hand count)
        recent_with_same_hands = [
            (name, conf) for name, conf, hands in self.prediction_history 
            if hands == num_hands
        ]
        
        if len(recent_with_same_hands) < 3:
            return mudra_name, confidence
        
        # Majority voting
        recent_names = [name for name, _ in recent_with_same_hands[-5:]]
        counter = Counter(recent_names)
        most_common = counter.most_common(1)[0]
        
        # If prediction appears in majority, use it
        if most_common[1] >= len(recent_with_same_hands[-5:]) // 2:
            confidences = [conf for name, conf in recent_with_same_hands if name == most_common[0]]
            avg_confidence = np.mean(confidences)
            return most_common[0], avg_confidence
        
        return mudra_name, confidence
    
    def draw_info_panel(self, frame, mudra_name, confidence, num_hands, top_predictions, fps):
        """Draw enhanced information panel"""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Top panel - larger for more info
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        
        # Bottom panel
        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
        
        # Right panel for top predictions
        if top_predictions:
            cv2.rectangle(overlay, (w-350, 150), (w, 450), (0, 0, 0), -1)
        
        # Blend
        alpha = 0.75
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Title
        cv2.putText(frame, "Bharatanatyam Mudra Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Model type indicator
        model_type = "Single Hand" if num_hands == 1 else "Double Hand" if num_hands == 2 else "No Hands"
        model_color = (100, 255, 100) if num_hands > 0 else (100, 100, 255)
        cv2.putText(frame, f"Mode: {model_type}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        # Detected mudra
        if num_hands > 0:
            if confidence > self.confidence_threshold:
                color = (0, 255, 0)
                status = "CONFIDENT"
            else:
                color = (0, 165, 255)
                status = "LOW CONFIDENCE"
            
            text = f"{mudra_name}"
            cv2.putText(frame, text, (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Status indicator
            cv2.putText(frame, status, (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(frame, "No hands detected - Show your mudra!", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # Confidence bar (only if hands detected)
        if num_hands > 0:
            bar_width = 300
            bar_x = w - bar_width - 20
            bar_y = 30
            bar_height = 20
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Confidence
            conf_width = int(bar_width * confidence)
            conf_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + conf_width, bar_y + bar_height), 
                         conf_color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 1)
            
            # Percentage
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                       (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Top predictions panel
        if top_predictions and num_hands > 0:
            y_offset = 165
            cv2.putText(frame, "Top Predictions:", (w - 330, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for i, (name, conf) in enumerate(top_predictions.items()):
                y_offset += 35
                # Color based on rank
                if i == 0:
                    pred_color = (0, 255, 0)
                elif i == 1:
                    pred_color = (100, 255, 100)
                else:
                    pred_color = (200, 200, 200)
                
                text = f"{i+1}. {name}"
                cv2.putText(frame, text, (w - 330, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
                
                # Mini confidence bar
                mini_bar_width = 300
                mini_bar_x = w - 330
                mini_bar_y = y_offset + 5
                mini_bar_height = 8
                
                cv2.rectangle(frame, (mini_bar_x, mini_bar_y),
                             (mini_bar_x + mini_bar_width, mini_bar_y + mini_bar_height),
                             (50, 50, 50), -1)
                
                mini_conf_width = int(mini_bar_width * conf)
                cv2.rectangle(frame, (mini_bar_x, mini_bar_y),
                             (mini_bar_x + mini_conf_width, mini_bar_y + mini_bar_height),
                             pred_color, -1)
                
                # Percentage
                cv2.putText(frame, f"{conf*100:.1f}%", 
                           (w - 330 + mini_bar_width + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                y_offset += 15
        
        # Instructions
        instructions = "Q: Quit | S: Screenshot | R: Reset | C: Clear history"
        cv2.putText(frame, instructions, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """Run enhanced live detection"""
        
        print("\n" + "="*70)
        print("Enhanced Bharatanatyam Mudra Detection System")
        print("="*70)
        print(f"Available models:")
        for model_type, info in self.model_info.items():
            print(f"  {model_type.title()} hand: {info['num_classes']} classes")
        print(f"\nCamera: {camera_id}")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("  R - Reset prediction history")
        print("  C - Clear screen")
        print("="*70 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # FPS calculation
        import time
        prev_time = time.time()
        fps = 0
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame, results = self.hand_tracker.find_hands(frame, draw=True)
            
            mudra_name = "No hands detected"
            confidence = 0.0
            top_predictions = {}
            num_hands = 0
            
            # If hands detected, predict
            if results.multi_hand_landmarks:
                features, num_hands = self.hand_tracker.extract_enhanced_features(results)
                
                if features is not None and num_hands > 0:
                    # Predict
                    mudra_name, confidence, top_predictions = self.predict(features, num_hands)
                    
                    # Smooth prediction
                    mudra_name, confidence = self.smooth_prediction(
                        mudra_name, confidence, num_hands
                    )
                    
                    # Draw bounding boxes
                    frame = self.hand_tracker.draw_bounding_box(frame, results)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Draw info panel
            frame = self.draw_info_panel(
                frame, mudra_name, confidence, num_hands, top_predictions, fps
            )
            
            # Display
            cv2.imshow('Enhanced Bharatanatyam Mudra Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                self.prediction_history.clear()
                print("Prediction history reset")
            elif key == ord('c') or key == ord('C'):
                # Clear terminal
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Screen cleared")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()
        
        print("\n✓ Detection stopped")

def main():
    detector = EnhancedMudraDetector(model_dir='models')
    detector.run(camera_id=0)

if __name__ == "__main__":
    main()