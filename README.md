# Fire Detection System

Real-time fire detection using YOLOv8 and webcam.

## Files

- `fire_detect.py` - Basic fire detection (uses COCO model)
- `fire_detect_advanced.py` - Advanced version with fire warnings
- `train_fire.py` - Train custom fire detection model
- `TRAINING_GUIDE.md` - Guide for training your own model
- `requirements.txt` - Python dependencies

## Quick Start

```bash
# Run basic detection
python fire_detect.py

# Run advanced detection (recommended)
python fire_detect_advanced.py
```

## Controls

- `q` - Quit
- `s` - Screenshot (advanced version)

## For Accurate Fire Detection

The default YOLOv8 model (COCO dataset) doesn't detect fire well. You need a custom trained model:

### Option 1: Download Pre-trained Fire Model
Search for "yolov8 fire detection model" and place `.pt` file in this folder.

### Option 2: Train Your Own (Best Accuracy)
1. Prepare dataset (see `TRAINING_GUIDE.md`)
2. Run: `python train_fire.py`
3. Use trained model: `runs/detect/fire_detection_model/weights/best.pt`

## Dataset Sources
- Roboflow Universe: "fire detection"
- Kaggle: "fire detection dataset"
- GitHub: "fire-detection-dataset"

## Requirements
- Python 3.8+
- Webcam
- See `requirements.txt`