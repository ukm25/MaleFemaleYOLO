# Real-time Gender Detection System

A Python application that uses YOLOv8 for person detection and DeepFace for gender classification to provide real-time gender detection from webcam feed.

## Features

- **Real-time Person Detection**: Uses YOLOv8 (nano/small/medium models) for fast and accurate person detection
- **Gender Classification**: Classifies each detected person's gender (Male/Female) using DeepFace
- **Live Webcam Feed**: Real-time processing with bounding boxes and gender labels
- **Performance Monitoring**: FPS counter and performance optimization
- **Modular Design**: Separate modules for detection, classification, and main application
- **Configurable Parameters**: Adjustable confidence thresholds and model sizes

## Requirements

- Python 3.8 or higher
- Webcam or camera device
- Minimum 4GB RAM (8GB recommended for better performance)
- GPU support recommended (but not required)

## Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import cv2, ultralytics, deepface; print('All dependencies installed successfully!')"
   ```

## Usage

### Basic Usage

Run the application with default settings:
```bash
python main.py
```

### Advanced Usage

```bash
# Use YOLOv8 small model for better accuracy
python main.py --model-size s

# Use fast classifier instead of DeepFace (faster but less accurate)
python main.py --fast-classifier

# Adjust detection confidence threshold
python main.py --confidence 0.6

# Adjust gender classification confidence threshold
python main.py --gender-confidence 0.7

# Use different camera device
python main.py --camera 1
```

### Command Line Arguments

- `--model-size`: YOLO model size ('n'=nano, 's'=small, 'm'=medium)
- `--fast-classifier`: Use fast classifier instead of DeepFace
- `--confidence`: YOLO detection confidence threshold (0.0-1.0)
- `--gender-confidence`: Gender classification confidence threshold (0.0-1.0)
- `--camera`: Camera device ID (default: 0)

### Controls

While the application is running:
- **'q'**: Quit the application
- **'r'**: Reset detections
- **'s'**: Save current frame as image

## Project Structure

```
MaleFemaleYOLO/
├── main.py                 # Main application
├── yolo_detector.py        # YOLO person detection module
├── gender_classifier.py    # Gender classification module
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Module Details

### yolo_detector.py
- **YOLOPersonDetector**: Handles YOLOv8 model loading and person detection
- **Key Methods**:
  - `detect_persons()`: Detect people in frames
  - `crop_person()`: Extract person regions
  - `draw_detections()`: Draw bounding boxes and labels

### gender_classifier.py
- **GenderClassifier**: DeepFace-based gender classification
- **FastGenderClassifier**: Lightweight alternative for better performance
- **Key Methods**:
  - `classify_gender()`: Classify gender from person image
  - `classify_multiple_persons()`: Batch gender classification

### main.py
- **RealTimeGenderDetector**: Main application class
- **Features**:
  - Real-time processing pipeline
  - Performance monitoring
  - User interface controls
  - Error handling and cleanup

## Performance Optimization

### For Better Speed:
1. Use YOLOv8n (nano) model: `--model-size n`
2. Use fast classifier: `--fast-classifier`
3. Increase confidence thresholds to reduce false positives
4. Ensure good lighting conditions
5. Use GPU acceleration if available

### For Better Accuracy:
1. Use YOLOv8s (small) or YOLOv8m (medium) model
2. Use DeepFace classifier (default)
3. Lower confidence thresholds for more detections
4. Ensure clear, well-lit images

## Troubleshooting

### Common Issues:

1. **Camera not found**:
   - Check if camera is connected and not used by other applications
   - Try different camera ID: `--camera 1`

2. **Low FPS performance**:
   - Use YOLOv8n model: `--model-size n`
   - Use fast classifier: `--fast-classifier`
   - Close other applications using the camera

3. **DeepFace installation issues**:
   - Install TensorFlow: `pip install tensorflow`
   - Use fast classifier as alternative: `--fast-classifier`

4. **YOLO model download issues**:
   - Ensure internet connection for first-time model download
   - Models are cached locally after first download

### Performance Tips:

- **Minimum 10-15 FPS**: Use YOLOv8n with fast classifier
- **Good accuracy**: Use YOLOv8s with DeepFace
- **Best performance**: Ensure good lighting and clear camera view

## Technical Details

### YOLOv8 Models:
- **YOLOv8n**: Fastest, ~6MB, good for real-time
- **YOLOv8s**: Balanced, ~22MB, better accuracy
- **YOLOv8m**: Most accurate, ~50MB, slower

### Gender Classification:
- **DeepFace**: Uses VGG-Face model, high accuracy
- **Fast Classifier**: Heuristic-based, very fast

### Detection Pipeline:
1. Capture frame from webcam
2. Run YOLO inference for person detection
3. Crop detected person regions
4. Classify gender for each person
5. Draw bounding boxes and labels
6. Display processed frame

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [DeepFace](https://github.com/serengil/deepface) for gender classification
- [OpenCV](https://opencv.org/) for computer vision utilities
