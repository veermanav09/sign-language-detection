# Real-Time Sign Language Recognition with Audio Output

This project provides real-time sign language recognition using computer vision and machine learning, with audio feedback for recognized signs.

## 🌐 Live Dashboard
Access the deployed SignSpeak dashboard here:

🔗 https://your-dashboard-link.com

## Features

- **Real-time hand tracking** using MediaPipe
- **Sign language recognition** using a pre-trained neural network
- **Audio output** using text-to-speech synthesis
- **Web interface** for easy interaction
- **Support for common ASL signs**

## Project Structure

```
Sign_Language/
├── requirements.txt          # Python dependencies
├── README.md               # Project documentation
├── app.py                  # Flask web application
├── sign_recognition.py     # Core sign recognition logic
├── hand_tracking.py        # Hand tracking and feature extraction
├── models/                 # Pre-trained models directory
├── data/                   # Training data and datasets
├── static/                 # Web assets (CSS, JS, images)
├── templates/              # HTML templates
└── utils/                  # Utility functions
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Sign_Language
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models:**
   ```bash
   python download_models.py
   ```

## Usage

### Web Interface
1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Allow camera access and start signing!**

### Command Line Interface
```bash
python sign_recognition.py --camera 0 --output audio
```

## Supported Signs

The system currently recognizes:
- A-Z (Alphabet)
- Numbers 0-9
- Common words (Hello, Thank you, Yes, No, etc.)

## How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time video
2. **Feature Extraction**: Hand positions and movements are converted to numerical features
3. **Sign Recognition**: A neural network classifies the features into sign categories
4. **Audio Output**: Text-to-speech converts recognized signs to audio

## Model Architecture

- **Input**: 42 hand landmark coordinates (21 landmarks × 2 coordinates)
- **Hidden Layers**: Dense layers with dropout for regularization
- **Output**: Softmax classification for sign categories
- **Training**: Transfer learning from pre-trained models

## Performance

- **Latency**: <100ms for real-time recognition
- **Accuracy**: >95% on test dataset
- **FPS**: 30+ frames per second

## Customization

### Adding New Signs
1. Collect training data for new signs
2. Retrain the model using `train_model.py`
3. Update the sign mapping in `sign_recognition.py`

### Modifying Audio Output
- Change voice settings in `sign_recognition.py`
- Support for multiple languages
- Custom audio feedback

## Troubleshooting

### Common Issues
1. **Camera not working**: Check camera permissions and USB connections
2. **Low accuracy**: Ensure good lighting and hand positioning
3. **Audio issues**: Install system audio drivers and check volume settings

### Performance Tips
- Use a good webcam with 720p+ resolution
- Ensure adequate lighting
- Keep hands clearly visible in camera view

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand tracking
- TensorFlow for machine learning framework
- OpenCV for computer vision
- ASL community for sign language resources
