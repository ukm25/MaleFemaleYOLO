"""
YOLO Person Detection Module

This module provides functionality to detect persons in images using YOLOv8.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

class YOLODetector:
    """
    A class for detecting persons in images using YOLOv8.
    
    This class handles loading the YOLO model and detecting persons
    in input images with bounding boxes and confidence scores.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO model file
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Load the YOLO model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
    
    def detect_persons(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            
        Returns:
            List[Tuple[int, int, int, int, float]]: List of detections as (x1, y1, x2, y2, confidence)
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        try:
            # Run YOLO inference
            results = self.model(image, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a person (class 0 in COCO dataset)
                        if class_id == 0 and confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            detections.append((x1, y1, x2, y2, confidence))
            
            return detections
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []
    
    def detect_persons_with_labels(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        Detect persons and return image with bounding boxes drawn.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            
        Returns:
            Tuple[np.ndarray, List]: (image_with_boxes, detections)
        """
        detections = self.detect_persons(image)
        
        # Create a copy of the image for drawing
        result_image = image.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result_image, detections
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

# Example usage
if __name__ == "__main__":
    # Test the YOLO detector
    detector = YOLODetector("yolov8n.pt")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect persons
    detections = detector.detect_persons(test_image)
    print(f"Found {len(detections)} persons")
    
    # Test with bounding boxes
    result_image, detections = detector.detect_persons_with_labels(test_image)
    print(f"Result image shape: {result_image.shape}")
