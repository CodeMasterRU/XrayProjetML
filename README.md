# Chest X-Ray Pneumonia Detection

This project uses deep learning models (MobileNetV2 and Vision Transformer) to detect pneumonia from chest X-ray images.

## Requirements
- Python 3.10
- TensorFlow
- Other dependencies listed in requirements.txt

## Installation
1. Create a virtual environment:
```bash
python -m venv chestxrayvenv
# Windows
chestxrayvenv\Scripts\activate

# Packages
pip install -r requirements.txt

# Creation of the models
python main.py

# Model tests
python test_result.py