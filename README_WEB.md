# 🎯 Gender & Age Detection Web App

A simple web application that automatically captures webcam images every 2 seconds and detects gender, age, race, and emotion using YOLO + DeepFace.

## 🚀 Features

- **Real-time Webcam**: Continuous webcam feed
- **Auto Capture**: Automatically captures images every 2 seconds
- **AI Detection**: Uses YOLOv8n + DeepFace for analysis
- **Multiple Attributes**: Detects gender, age, race, and emotion
- **Web UI**: Simple, clean interface
- **Console Logging**: Results displayed in browser console

## 📁 Project Structure

```
MaleFemaleYOLO/
├── app.py                 # Flask API server
├── index.html            # Web UI (TypeScript/JavaScript)
├── yolo_detector.py      # YOLO person detection
├── gender_classifier.py  # DeepFace analysis
├── requirements.txt      # Python dependencies
└── README_WEB.md        # This file
```

## 🛠️ Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python app.py
```
The API will be available at: `http://localhost:5000`

### 3. Open the Web UI
Open `index.html` in your web browser or serve it with a local server:
```bash
# Using Python's built-in server
python -m http.server 8000
```
Then open: `http://localhost:8000`

## 🎮 Usage

1. **Open the Web UI** in your browser
2. **Allow webcam access** when prompted
3. **Click "Start Detection"** to begin auto-capturing
4. **View results** in the UI and browser console
5. **Click "Stop Detection"** to stop

## 📊 API Endpoints

### POST /api/detect
Detects gender, age, race, and emotion from an image.

**Request:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "gender": "Nữ",
      "gender_confidence": 0.85,
      "age": 25,
      "race": "Châu Á",
      "race_confidence": 0.78,
      "emotion": "Vui vẻ",
      "emotion_confidence": 0.92,
      "bbox": [100, 50, 200, 300],
      "detection_confidence": 0.95
    }
  ],
  "processing_time": 0.15,
  "timestamp": 1697123456.789
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": 1697123456.789
}
```

## ⚡ Performance

- **Capture Interval**: 2 seconds
- **Processing Time**: ~100-200ms per image
- **Model**: YOLOv8n (fast) + DeepFace (accurate)
- **Image Size**: 640x480 pixels
- **Compression**: JPEG 80% quality

## 🔧 Configuration

### Change Capture Interval
Edit the interval in `index.html`:
```javascript
// Change from 2000ms (2s) to 3000ms (3s)
this.captureInterval = setInterval(() => {
    this.captureAndDetect();
}, 3000);
```

### Change Model
Edit the model in `app.py`:
```python
# Change from yolov8n.pt to yolov8s.pt for better accuracy
detector = YOLODetector(model_path="yolov8s.pt")
```

## 🐛 Troubleshooting

### Webcam Not Working
- Ensure browser has camera permissions
- Try refreshing the page
- Check if another application is using the camera

### API Connection Error
- Ensure Flask server is running on port 5000
- Check browser console for CORS errors
- Verify `flask-cors` is installed

### Slow Processing
- Reduce image quality in `index.html` (line with `toDataURL`)
- Use YOLOv8n instead of YOLOv8s
- Increase capture interval

## 🎯 Next Steps

- Add face detection visualization
- Implement real-time video streaming
- Add database storage for results
- Create user authentication
- Deploy to cloud platform

## 📝 Notes

- Results are logged to browser console for debugging
- Images are compressed before sending to reduce bandwidth
- The app works best with good lighting conditions
- Multiple people in frame will be detected separately
