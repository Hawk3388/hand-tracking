# Hand Tracking 🖐️

Control your computer with just your hands.

## Features

- **Hand gesture recognition** for mouse control
- **Two modes**: GUI window or background operation
- **ASL recognition model** (American Sign Language) support
- **Multi-monitor support**
- **Standalone executables** - no Python installation required

## System Requirements

- Windows 10/11
- Webcam
- For Python users: Python 3.8-3.11

## Quick Start

### Option 1: Pre-built Executables (Recommended)

1. Download the .exe file from the latest release on [GitHub Releases](https://github.com/Hawk3388/hand-tracking/releases)
2. Run `hand-tracking.exe` for GUI mode or `hand-tracking-no-window.exe` for background mode

### Option 2: Run with Python

```bash
git clone https://github.com/Hawk3388/hand-tracking.git
cd hand-tracking
pip install -r requirements.txt
python src/hand-tracking.py          # GUI mode
python src/hand-tracking-no-window.py # Background mode
```

### Option 3: Build Your Own Executables

```bash
pip install pyinstaller==6.12.0
python src/build_exe.py
```

This creates two executables in the `dist/` folder.

## Usage

- **Mouse Control**: Move your hand to control the cursor
- **Click**: Make a fist to click
- **Mode Switch**: Thumb + Middle finger to toggle between mouse and ASL modes
- **Multi-monitor**: Move hand to screen edges to switch monitors

## Controls

- **Mouse Mode**: Hand movements control cursor, fist = click
- **ASL Mode**: Recognizes American Sign Language gestures
- **Exit**: Press 'q' in GUI mode or Ctrl+C in console mode

## Credits

Special thanks to:

- [ASL Now Fingerspelling Dataset](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) for the ASL training data
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=de) by Google for the hand tracking model

## License

MIT License - see [LICENSE](LICENSE) file
