# 🤖 Hand Gesture Cursor Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Control your computer with hand gestures! A complete mouse replacement system using computer vision.**

[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Click the image above to watch the demo video*

</div>

## 🚀 Features

### ✨ **Complete Mouse Replacement**
- **Cursor Movement**: Navigate with your index finger
- **Left Click**: Bend your middle finger
- **Right Click**: Bend your ring finger  
- **Double Click**: Two quick middle finger bends
- **Scroll Control**: Move thumb up/down to scroll

### 🎵 **Volume Control**
- **Volume Adjustment**: Control volume with thumb-index distance
- **Volume Lock**: Raise pinkie to lock current volume
- **Visual Feedback**: Real-time volume percentage display

### 👥 **Multi-Hand Support**
- **Left/Right Hand Detection**: Automatic hand type recognition
- **Color-Coded Feedback**: Green for left hand, Red for right hand
- **Flexible Usage**: Switch between hands seamlessly

### 🎯 **Precision Control**
- **Control Box**: Centered 1/3 screen area for precise movement
- **Smooth Movement**: Advanced smoothing algorithms
- **Visual Feedback**: Enhanced cursor with glow effects

## 📸 Screenshots

<div align="center">

### Main Interface
![Main Interface](images/main_interface.png)
*The main control interface showing hand detection and gesture instructions*

### Gesture Recognition
![Gesture Recognition](images/gesture_recognition.png)
*Real-time gesture recognition with visual feedback*

### Volume Control
![Volume Control](images/volume_control.png)
*Volume control interface with distance-based adjustment*

</div>

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows 10/11 (for audio control features)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hand-gesture-control.git
   cd hand-gesture-control
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python hand_gesture_control.py
   ```

### Detailed Installation Guide

<details>
<summary>Click to expand detailed installation steps</summary>

#### Step 1: Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv hand_gesture_env
source hand_gesture_env/bin/activate  # On Windows: hand_gesture_env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 2: System Requirements
- **Webcam**: Any USB webcam or built-in camera
- **Lighting**: Ensure good lighting for hand detection
- **Space**: Clear area in front of camera for hand gestures

#### Step 3: Audio Setup (Windows)
The volume control feature requires Windows audio system access:
```bash
# Install audio dependencies
pip install pycaw comtypes
```

#### Step 4: Test Installation
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Test hand detection
python -c "import mediapipe as mp; print('MediaPipe OK')"
```

</details>

## 🎮 Usage Guide

### Basic Gestures

| Gesture | Action | How to Perform |
|---------|--------|----------------|
| **🖱️ Move Cursor** | Navigate | Move index finger within the yellow control box |
| **👆 Left Click** | Click | Bend middle finger down |
| **🖱️ Right Click** | Right-click | Bend ring finger down |
| **👆👆 Double Click** | Double-click | Two quick middle finger bends |
| **📜 Scroll Up** | Scroll up | Move thumb up (high position) |
| **📜 Scroll Down** | Scroll down | Move thumb down (low position) |
| **🔊 Volume** | Adjust volume | Change distance between thumb and index finger |
| **🔒 Volume Lock** | Lock volume | Raise pinkie finger |

### Advanced Usage

<details>
<summary>Click to expand advanced usage tips</summary>

#### Precision Control
- **Control Box**: The yellow rectangle is your active area (1/3 of screen)
- **Smooth Movement**: Cursor movement is smoothed for better control
- **Speed Adjustment**: Modify `cursor_speed` in the code for different sensitivity

#### Volume Control
- **Distance Method**: Thumb and index finger distance controls volume
- **Lock Feature**: Raise pinkie to lock current volume level
- **Range**: 30% to 100% volume range

#### Multi-Hand Support
- **Automatic Detection**: System detects left or right hand automatically
- **Color Coding**: Green landmarks for left hand, red for right hand
- **Flexible Switching**: Use either hand or both hands

</details>

## 🔧 Configuration

### Customization Options

You can modify these parameters in `hand_gesture_control.py`:

```python
# Cursor Control
cursor_speed = 3          # Higher = faster cursor movement
smoothening = 8           # Higher = smoother movement

# Click Control
click_cooldown = 0.3      # Seconds between clicks
double_click_threshold = 0.5  # Seconds for double-click detection

# Scroll Control
scroll_cooldown = 0.1     # Seconds between scroll actions

# Volume Control
min_dist, max_dist = 20, 200  # Distance range for volume control
```

### Advanced Configuration

<details>
<summary>Click to expand advanced configuration</summary>

#### Camera Settings
```python
# Camera resolution (modify in VideoCapture)
cap = cv2.VideoCapture(0)  # Camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
```

#### Hand Detection Settings
```python
# MediaPipe hand detection parameters
hands = mp_hands.Hands(
    max_num_hands=2,                    # Maximum hands to detect
    min_detection_confidence=0.7,       # Detection confidence threshold
    min_tracking_confidence=0.5         # Tracking confidence threshold
)
```

#### Performance Tuning
```python
# For better performance on slower systems
smoothening = 10        # Higher = smoother but slower
cursor_speed = 2         # Lower = more precise but slower
```

</details>

## 🏗️ Architecture

### System Components

```
Hand Gesture Controller
├── Camera Input (OpenCV)
├── Hand Detection (MediaPipe)
├── Gesture Recognition
│   ├── Cursor Movement
│   ├── Click Detection
│   ├── Scroll Control
│   └── Volume Control
├── System Integration
│   ├── Mouse Control (PyAutoGUI)
│   └── Audio Control (PyCaw)
└── Visual Feedback (OpenCV)
```

### Key Classes and Methods

<details>
<summary>Click to expand technical details</summary>

#### HandGestureController Class
- **`__init__()`**: Initialize all components
- **`setup_audio()`**: Configure audio system
- **`handle_cursor_control()`**: Process cursor movement
- **`handle_click_controls()`**: Process click events
- **`handle_scroll_control()`**: Process scroll events
- **`handle_volume_control()`**: Process volume adjustment
- **`run()`**: Main application loop

#### Key Methods Explained
```python
def handle_cursor_control(self, frame, hand_landmarks, w, h):
    """Handle cursor movement with control box mapping"""
    # Maps finger position to screen coordinates
    # Applies smoothing for better control
    # Provides visual feedback

def handle_click_controls(self, frame, hand_landmarks, w, h):
    """Handle all click types with gesture recognition"""
    # Middle finger = left click
    # Ring finger = right click
    # Double-click detection with timing
```

</details>

## 🐛 Troubleshooting

### Common Issues

<details>
<summary>Click to expand troubleshooting guide</summary>

#### Camera Issues
```bash
# Problem: Camera not detected
# Solution: Check camera permissions and USB connection
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### Hand Detection Issues
```bash
# Problem: Hands not detected
# Solution: Check lighting and hand position
# - Ensure good lighting
# - Keep hand within camera frame
# - Avoid cluttered background
```

#### Audio Control Issues
```bash
# Problem: Volume control not working
# Solution: Check Windows audio system
# - Ensure Windows 10/11
# - Check audio device permissions
# - Verify pycaw installation
```

#### Performance Issues
```bash
# Problem: Laggy performance
# Solution: Optimize settings
# - Reduce camera resolution
# - Increase smoothening value
# - Close other applications
```

</details>

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Camera not found` | No camera detected | Check camera connection |
| `Audio setup failed` | Audio system issue | Check Windows audio |
| `Hand not detected` | Poor lighting/position | Improve lighting |
| `Cursor stuck` | High sensitivity | Reduce cursor_speed |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/hand-gesture-control.git
cd hand-gesture-control
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black hand_gesture_control.py

# Lint code
flake8 hand_gesture_control.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** - For hand landmark detection
- **OpenCV** - For computer vision processing
- **PyAutoGUI** - For mouse and keyboard automation
- **PyCaw** - For Windows audio control

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hand-gesture-control/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hand-gesture-control/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] Voice commands integration
- [ ] Multi-monitor support
- [ ] Custom gesture training
- [ ] Mobile app companion
- [ ] Gesture recording and playback
- [ ] Accessibility features
- [ ] Machine learning optimization

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Your Name](https://github.com/yourusername)

</div>
