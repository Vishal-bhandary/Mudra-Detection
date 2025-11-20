"""
Enhanced landmark extraction with better feature engineering
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
import pickle
from collections import defaultdict

class EnhancedLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower for better detection
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_distances(self, landmarks):
        """Calculate distances between key points"""
        distances = []
        # Distance from wrist to each fingertip
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip in fingertips:
            dist = np.sqrt(
                (landmarks[tip][0] - wrist[0])**2 + 
                (landmarks[tip][1] - wrist[1])**2
            )
            distances.append(dist)
        
        # Distance between adjacent fingertips
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
        
        # Define finger chains: [[base, middle, tip], ...]
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
                
                # Calculate angle
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        return angles
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist and hand size"""
        landmarks = np.array(landmarks)
        
        # Use wrist as origin
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        # Calculate hand size (max distance from wrist)
        distances = np.sqrt(np.sum(landmarks**2, axis=1))
        max_dist = np.max(distances) + 1e-6
        
        # Normalize by hand size
        landmarks = landmarks / max_dist
        
        return landmarks.flatten()
    
    def extract_enhanced_features(self, image_path):
        """Extract comprehensive hand features"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        all_features = []
        hand_info = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract raw landmarks
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
            
            # Get handedness
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
            else:
                handedness = "Unknown"
            
            # Combine all features
            features = {
                'landmarks': norm_landmarks.tolist(),
                'distances': distances,
                'angles': angles,
                'handedness': handedness
            }
            
            all_features.append(features)
            hand_info.append(handedness)
        
        return {
            'features': all_features,
            'num_hands': len(all_features),
            'hand_info': hand_info
        }
    
    def draw_landmarks(self, image_path, output_path):
        """Draw landmarks on image and save"""
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            cv2.imwrite(output_path, image)
            return True
        return False
    
    def close(self):
        self.hands.close()

def process_dataset_enhanced(dataset_path, output_dir, visualize_samples=100):
    """Process entire dataset with enhanced features"""
    
    extractor = EnhancedLandmarkExtractor()
    
    # Create output directories
    landmarks_dir = os.path.join(output_dir, 'landmarks')
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Separate datasets for single and double hand
    dataset = {
        'single_hand': [],
        'double_hand': []
    }
    
    # Track statistics
    mudra_stats = defaultdict(lambda: {'single': 0, 'double': 0, 'failed': 0})
    failed_images = []
    sample_count = 0
    
    # Get all mudra folders
    mudra_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
    
    print(f"Found {len(mudra_folders)} mudra classes")
    
    # Process each mudra folder
    for mudra_name in tqdm(mudra_folders, desc="Processing mudras"):
        mudra_path = os.path.join(dataset_path, mudra_name)
        
        image_files = [f for f in os.listdir(mudra_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"  {mudra_name}", leave=False):
            img_path = os.path.join(mudra_path, img_file)
            
            # Extract features
            result = extractor.extract_enhanced_features(img_path)
            
            if result is None:
                failed_images.append(img_path)
                mudra_stats[mudra_name]['failed'] += 1
                continue
            
            # Determine if single or double hand
            num_hands = result['num_hands']
            hand_type = 'single_hand' if num_hands == 1 else 'double_hand'
            
            # Store data
            data_entry = {
                'mudra_name': mudra_name,
                'image_path': img_path,
                'features': result['features'],
                'num_hands': num_hands,
                'hand_info': result['hand_info']
            }
            
            dataset[hand_type].append(data_entry)
            mudra_stats[mudra_name][hand_type.replace('_hand', '')] += 1
            
            # Visualize some samples
            if sample_count < visualize_samples:
                vis_path = os.path.join(
                    visualizations_dir, 
                    f"{mudra_name}_{img_file}"
                )
                if extractor.draw_landmarks(img_path, vis_path):
                    sample_count += 1
    
    extractor.close()
    
    # Save processed data
    print("\n\nSaving processed data...")
    
    # Save as pickle
    with open(os.path.join(landmarks_dir, 'dataset_enhanced.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    
    # Create comprehensive metadata
    mudra_classes_single = list(set([d['mudra_name'] for d in dataset['single_hand']]))
    mudra_classes_double = list(set([d['mudra_name'] for d in dataset['double_hand']]))
    
    metadata = {
        'total_single_hand': len(dataset['single_hand']),
        'total_double_hand': len(dataset['double_hand']),
        'total_images': len(dataset['single_hand']) + len(dataset['double_hand']),
        'failed_images': len(failed_images),
        'mudra_classes_single': sorted(mudra_classes_single),
        'mudra_classes_double': sorted(mudra_classes_double),
        'num_classes_single': len(mudra_classes_single),
        'num_classes_double': len(mudra_classes_double),
        'mudra_statistics': dict(mudra_stats)
    }
    
    with open(os.path.join(landmarks_dir, 'metadata_enhanced.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save detailed statistics
    print(f"\n{'='*70}")
    print(f"Processing Complete!")
    print(f"{'='*70}")
    print(f"Single hand mudras: {metadata['total_single_hand']} images, {metadata['num_classes_single']} classes")
    print(f"Double hand mudras: {metadata['total_double_hand']} images, {metadata['num_classes_double']} classes")
    print(f"Total processed: {metadata['total_images']}")
    print(f"Failed: {metadata['failed_images']}")
    
    print(f"\n{'='*70}")
    print("Mudra Distribution:")
    print(f"{'='*70}")
    for mudra, stats in sorted(mudra_stats.items()):
        print(f"{mudra:30s} | Single: {stats['single']:4d} | Double: {stats['double']:4d} | Failed: {stats['failed']:3d}")
    
    print(f"\nData saved to: {landmarks_dir}")
    print(f"Visualizations saved to: {visualizations_dir}")
    
    # Save failed images list
    if failed_images:
        with open(os.path.join(landmarks_dir, 'failed_images.txt'), 'w') as f:
            f.write('\n'.join(failed_images))
    
    return dataset, metadata

if __name__ == "__main__":
    DATASET_PATH = "Bharatanatyam-Mudra-Dataset-master/Images"
    OUTPUT_DIR = "data/processed"
    
    dataset, metadata = process_dataset_enhanced(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        visualize_samples=100
    )
    
    print("\n✓ Enhanced landmark extraction complete!")
    print("Next step: Run 2_train_model.py")