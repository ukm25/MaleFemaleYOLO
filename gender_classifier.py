"""
Gender Classification Module

This module provides functionality to classify gender from person images using DeepFace.
It uses a lightweight approach to ensure real-time performance.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import warnings

# Suppress DeepFace warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Please install it using: pip install deepface")


class GenderClassifier:
    """
    A class for classifying gender from person images using DeepFace.
    
    This class handles loading the gender classification model and
    predicting gender from cropped person images.
    """
    
    def __init__(self, model_name: str = 'VGG-Face', confidence_threshold: float = 0.6):
        """
        Initialize the gender classifier with extended analysis capabilities.
        
        Args:
            model_name (str): DeepFace model to use for analysis
            confidence_threshold (float): Minimum confidence for predictions
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is not available. Please install it first.")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.gender_labels = ['Woman', 'Man']  # DeepFace gender labels
        
        print(f"Initializing advanced face analyzer with model: {model_name}")
        
        # Test the model with a dummy image to ensure it's loaded
        try:
            # Create a small test image
            test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            _ = DeepFace.analyze(test_img, actions=['gender', 'age', 'race', 'emotion'], 
                               enforce_detection=False, silent=True)
            print("Advanced face analyzer initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize face analyzer: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for gender classification.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to a standard size for better performance
        # DeepFace can handle various sizes, but smaller images are faster
        height, width = rgb_image.shape[:2]
        if height < 100 or width < 100:
            # If image is too small, resize it
            rgb_image = cv2.resize(rgb_image, (224, 224))
        
        return rgb_image
    
    def analyze_face(self, person_image: np.ndarray) -> dict:
        """
        Analyze face for gender, age, race, and emotion.
        
        Args:
            person_image (np.ndarray): Cropped image of a person (BGR format)
            
        Returns:
            dict: Analysis results with gender, age, race, emotion
        """
        if not DEEPFACE_AVAILABLE:
            return {
                'gender': 'Không xác định',
                'gender_confidence': 0.0,
                'age': 0,
                'race': 'Không xác định',
                'race_confidence': 0.0,
                'emotion': 'Không xác định',
                'emotion_confidence': 0.0
            }
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(person_image)
            
            # Analyze all attributes using DeepFace
            result = DeepFace.analyze(
                processed_image, 
                actions=['gender', 'age', 'race', 'emotion'], 
                enforce_detection=False,  # Don't enforce face detection
                silent=True  # Suppress output
            )
            
            # Extract results
            if isinstance(result, list):
                result = result[0]  # Take first result if multiple faces
            
            # Process gender
            gender_data = result.get('gender', {})
            dominant_gender = result.get('dominant_gender', 'Unknown')
            
            if isinstance(gender_data, dict):
                woman_score = gender_data.get('Woman', 0)
                man_score = gender_data.get('Man', 0)
                total_score = woman_score + man_score
                
                if total_score > 0:
                    gender_confidence = max(woman_score, man_score) / total_score
                else:
                    gender_confidence = 0.5
            else:
                gender_confidence = 0.5
            
            # Convert gender to Vietnamese
            if dominant_gender in ['Woman', 'Female']:
                gender = "Nữ"
            elif dominant_gender in ['Man', 'Male']:
                gender = "Nam"
            else:
                gender = "Không xác định"
            
            # Process age
            age = result.get('age', 0)
            
            # Process race
            race_data = result.get('race', {})
            dominant_race = result.get('dominant_race', 'Unknown')
            
            if isinstance(race_data, dict):
                race_scores = list(race_data.values())
                if race_scores:
                    race_confidence = max(race_scores) / sum(race_scores)
                else:
                    race_confidence = 0.5
            else:
                race_confidence = 0.5
            
            # Convert race to Vietnamese
            race_map = {
                'asian': 'Châu Á',
                'indian': 'Ấn Độ',
                'black': 'Châu Phi',
                'white': 'Châu Âu',
                'middle eastern': 'Trung Đông',
                'latino hispanic': 'Mỹ Latinh'
            }
            race = race_map.get(dominant_race.lower(), 'Không xác định')
            
            # Process emotion
            emotion_data = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
            
            if isinstance(emotion_data, dict):
                emotion_scores = list(emotion_data.values())
                if emotion_scores:
                    emotion_confidence = max(emotion_scores) / sum(emotion_scores)
                else:
                    emotion_confidence = 0.5
            else:
                emotion_confidence = 0.5
            
            # Convert emotion to Vietnamese
            emotion_map = {
                'angry': 'Tức giận',
                'disgust': 'Ghê tởm',
                'fear': 'Sợ hãi',
                'happy': 'Vui vẻ',
                'sad': 'Buồn bã',
                'surprise': 'Ngạc nhiên',
                'neutral': 'Bình thường'
            }
            emotion = emotion_map.get(dominant_emotion.lower(), 'Không xác định')
            
            return {
                'gender': gender,
                'gender_confidence': gender_confidence,
                'age': age,
                'race': race,
                'race_confidence': race_confidence,
                'emotion': emotion,
                'emotion_confidence': emotion_confidence
            }
                
        except Exception as e:
            print(f"Error in face analysis: {e}")
            return {
                'gender': 'Không xác định',
                'gender_confidence': 0.0,
                'age': 0,
                'race': 'Không xác định',
                'race_confidence': 0.0,
                'emotion': 'Không xác định',
                'emotion_confidence': 0.0
            }
    
    def classify_gender(self, person_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify gender from a person image (backward compatibility).
        
        Args:
            person_image (np.ndarray): Cropped image of a person (BGR format)
            
        Returns:
            Tuple[str, float]: Gender prediction and confidence score
        """
        result = self.analyze_face(person_image)
        return result['gender'], result['gender_confidence']
    
    def classify_multiple_persons(self, person_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classify gender for multiple person images.
        
        Args:
            person_images (List[np.ndarray]): List of cropped person images
            
        Returns:
            List[Tuple[str, float]]: List of gender predictions and confidence scores
        """
        results = []
        
        for person_image in person_images:
            gender, confidence = self.classify_gender(person_image)
            results.append((gender, confidence))
        
        return results
    
    def is_valid_person_image(self, image: np.ndarray) -> bool:
        """
        Check if the image is valid for gender classification.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None or image.size == 0:
            return False
        
        height, width = image.shape[:2]
        return height >= 50 and width >= 50
    
    def _classify_gender_simple(self, person_image: np.ndarray) -> Tuple[str, float]:
        """
        Simple heuristic-based gender classification as fallback.
        
        Args:
            person_image (np.ndarray): Cropped image of a person
            
        Returns:
            Tuple[str, float]: Gender prediction and confidence score
        """
        try:
            height, width = person_image.shape[:2]
            
            # Ensure the crop is large enough
            if height < 50 or width < 50:
                return "Không xác định", 0.0
            
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
            
            # Analyze color/brightness
            mean_brightness = np.mean(gray)
            
            # Analyze shape (aspect ratio)
            aspect_ratio = width / height
            
            # Simple heuristic logic
            if mean_brightness > 130:  # Brighter images might lean towards 'Nữ'
                gender = "Nữ"
                confidence = 0.65
            elif mean_brightness < 100:  # Darker images might lean towards 'Nam'
                gender = "Nam" 
                confidence = 0.65
            else:
                # If brightness is ambiguous, use aspect ratio
                if aspect_ratio > 0.8:  # Wider face
                    gender = "Nam"
                    confidence = 0.55
                else:  # Taller face
                    gender = "Nữ"
                    confidence = 0.55
            
            return gender, confidence
            
        except Exception as e:
            return "Không xác định", 0.0


# Example usage
if __name__ == "__main__":
    # Test the gender classifier
    classifier = GenderClassifier()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Classify gender
    gender, confidence = classifier.classify_gender(test_image)
    print(f"Predicted gender: {gender} (confidence: {confidence:.2f})")