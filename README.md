# Face recognition with OpenCV

This repo is a personal project and the images references memes! (right now the program is using the hamster reactions from tiktok)

## Instructions
I installed OpenCV into a venv (virtual environment) but it is up to you if you want to!
If you do, create a venv:
    
`python -m venv venv`

Activate the venv:
    
`.\venv\Scripts\activate` (for windows)
`source venv/bin/activate` (for macos/linux)

Install OpenCV:

`pip install opencv-python`

## How it works
Currently, the data capturer is set up (collect_emotions.py)
When you run that file, it will use your webcam to capture images and it will automatically create a location (dataset) and sort the images into them (train/test)!

if this is not working properly: (in train_emotion_model.py)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
