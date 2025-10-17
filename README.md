# 🎯 Gender & Age Detection Web App

A real-time web application that automatically captures webcam images and detects gender, age, race, and emotion using YOLO + DeepFace.

## 🚀 Features

- **Real-time Webcam**: Continuous webcam feed with horizontal layout
- **Auto Capture**: Automatically captures images every 2 seconds
- **AI Detection**: Uses YOLOv8n + DeepFace for accurate analysis
- **Multiple Attributes**: Detects gender, age, race, and emotion
- **Responsive Design**: Works on desktop and mobile devices
- **Flask API**: RESTful API for image processing
- **Console Logging**: Results displayed in browser console

## 📁 Project Structure

```
MaleFemaleYOLO/
├── app.py                 # Flask API server
├── index.html            # Web UI with horizontal layout
├── yolo_detector.py      # YOLO person detection
├── gender_classifier.py  # DeepFace analysis
├── main.py              # YOLOv8n + DeepFace combo
├── main_yolov8s.py      # YOLOv8s + DeepFace combo
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── README_WEB.md       # Web app documentation
├── yolov8n.pt         # YOLO nano model (6MB)
└── yolov8s.pt         # YOLO small model (22MB)
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/ukm25/MaleFemaleYOLO.git
cd MaleFemaleYOLO
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start API Server
```bash
python app.py
```
The API will be available at: `http://localhost:5001`

### 4. Open Web UI
Open `index.html` in your browser or serve it locally:
```bash
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

## ⚡ Performance

- **Capture Interval**: 2 seconds
- **Processing Time**: ~100-200ms per image
- **Model**: YOLOv8n (fast) + DeepFace (accurate)
- **Image Size**: 640x480 pixels
- **Compression**: JPEG 80% quality

## 🎨 UI Layout

The application features a horizontal layout:
- **Left Panel**: Webcam feed and controls
- **Right Panel**: Detection results
- **Responsive**: Automatically adjusts for mobile devices

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
- Ensure Flask server is running on port 5001
- Check browser console for CORS errors
- Verify `flask-cors` is installed

### Slow Processing
- Reduce image quality in `index.html`
- Use YOLOv8n instead of YOLOv8s
- Increase capture interval

## 🎯 Use Cases

- **Security Systems**: Real-time person monitoring
- **Retail Analytics**: Customer demographic analysis
- **Healthcare**: Patient monitoring and analysis
- **Education**: Student engagement tracking
- **Research**: Human behavior studies

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
- **Heroku**: Use Procfile and requirements.txt
- **Railway**: Direct deployment from GitHub
- **AWS**: EC2 with Docker container
- **Google Cloud**: App Engine or Cloud Run

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For support, email quang251199@gmail.com or create an issue on GitHub.

## 🎉 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **DeepFace**: Serengil for facial analysis
- **Flask**: Web framework
- **OpenCV**: Computer vision library

---

**Made with ❤️ by ukm25**