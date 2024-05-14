from ultralytics import YOLO
import cv2
import path_routing
import os
import numpy as np
import matplotlib.pyplot as plt

#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
def train_the_model():
#для тренировки модели
    model = YOLO('yolov8n-seg.pt')
    results = model.train(data=os.path.join(path_routing.PROJECT_DIR,'config.yaml'), epochs=100, imgsz=640) #Файл .yaml должен содержать информацию о данных(где хранятся)
#train_the_model()

def example_case(path_to_image, path_to_model,to_csv = False,save_image = False):
    if not to_csv:
        model_path = path_to_model #после тренировки будет сформирован файл с информацией, оттуда берем папку weight (разделители - "\\")
        image_path = path_to_image #путь до изображения, которое хотим прогнать серез модель (разделители  - "\\")
        img = cv2.imread(image_path)
        overlay = np.zeros_like(img, dtype=np.uint8)
        H, W, _ = img.shape
        model = YOLO(model_path)
        results = model.predict(img,imgsz = 640,box = False,show_labels = False,line_width = 2) #прогнозирование записывается в папку 'runs'
        for result in results: #проход по всем результатам
            if result.masks is not None:
                for j, mask in enumerate(result.masks.data): #Выборка масок изображения
                    mask = mask.cpu().numpy() * 255  # Move tensor to CPU and convert to numpy array
                    mask = cv2.resize(mask, (W, H))
                    #перенесение масок в отдельный массив
                    overlay[:, :, 0] = overlay[:, :, 0]+mask
                    overlay[:, :, 1] = overlay[:, :, 0]+mask
                    overlay[:, :, 2] = overlay[:, :, 0]+mask
        res = cv2.addWeighted(img,0.8,overlay,0.5,0) #наложение изображений
        if not save_image:
            plt.imshow(overlay)
            #plt.imshow(res) #если есть желание сразу визуализировать - раскоментить
            plt.show() # ^
        else:
            #cv2.imwrite('result.png',overlay) #записывается в папку 'app'
            return res,overlay
    else:
        pass #тут нужно вернуть объем



#example_case()