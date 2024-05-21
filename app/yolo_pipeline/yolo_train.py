from ultralytics import YOLO
import cv2
from config.settings import path_routing
import os
import numpy as np
import matplotlib.pyplot as plt

#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
def train_the_model():
#для тренировки модели
    model = YOLO('yolov8n-seg.pt')
    results = model.train(data=os.path.join(path_routing.project_dir, 'config.yaml'), epochs=100, imgsz=640) #Файл .yaml должен содержать информацию о данных(где хранятся)
#train_the_model()