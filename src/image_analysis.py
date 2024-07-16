import cv2
import pytesseract
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def object_identification(image_path):
    img = Image.open(image_path)
    results = model(img)
    results.show()
    return results.pandas().xyxy

def color_identification(image_path):
    img = cv2.imread(image_path)
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers

def position_extraction(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def character_recognition(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text
