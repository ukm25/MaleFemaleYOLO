#!/usr/bin/env python3
"""
Main script for Fast Accurate Gender Detection
YOLOv8n + DeepFace (Fast but still accurate)

This version uses YOLOv8n for faster detection but keeps DeepFace for accuracy.
"""

import cv2
import time
import numpy as np
from yolo_detector import YOLODetector
from gender_classifier import GenderClassifier

class FastAccurateGenderDetector:
    def __init__(self):
        """Initialize the fast accurate gender detection system."""
        print("🎯 Khởi tạo hệ thống YOLOv8n + DeepFace (NHANH NHƯNG CHÍNH XÁC)...")
        
        # Initialize YOLO detector with nano model for speed
        self.detector = YOLODetector(model_path="yolov8n.pt")
        print("✅ YOLOv8n model loaded!")
        
        # Initialize gender classifier with balanced settings
        self.gender_classifier = GenderClassifier(
            model_name='VGG-Face',
            confidence_threshold=0.65  # Balanced threshold
        )
        print("✅ Gender classifier loaded!")
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        print("✅ Hệ thống đã sẵn sàng!")
        print("🎯 Dự kiến: 20-25 FPS, độ chính xác: 75-80%")
        print("🎯 Bắt đầu xác định giới tính...")
        print("⌨️  Nhấn 'q' để thoát, 's' để lưu ảnh")
    
    def process_frame(self, frame):
        """Process a single frame for gender detection."""
        # Detect persons
        detections = self.detector.detect_persons(frame)
        
        # Process each detection
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Crop person from frame
            person_crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if person_crop.shape[0] < 50 or person_crop.shape[1] < 50:
                continue
            
            # Classify gender
            gender, gender_confidence = self.gender_classifier.classify_gender(person_crop)
            
            # Show predictions with reasonable confidence
            if gender_confidence >= 0.55:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label
                label = f"{gender} ({gender_confidence:.2f})"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Run the gender detection system."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở camera!")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Không thể đọc frame từ camera!")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate and display FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed_time = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed_time
                
                # Display FPS
                fps_text = f"FPS: {self.fps:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Fast Accurate Gender Detection", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"gender_detection_fast_accurate_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"✅ Đã lưu ảnh: {filename}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Dừng bởi người dùng")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Đã dọn dẹp tài nguyên!")

def main():
    """Main function."""
    try:
        detector = FastAccurateGenderDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Hãy kiểm tra camera và dependencies")

if __name__ == "__main__":
    main()
