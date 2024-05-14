# for convinience
# put all needed functions in here and execute main.py
import csv
import glob

from pydicom.errors import InvalidDicomError

# raw_data_processing()
import marked_mri_processing
import yolo_train
import path_routing
import os
import pydicom
import numpy as np
import dicom2nifti
import shutil
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import csv


def folder_creator():
    name = 'research'
    path_to_folder_research = os.path.join(path_routing.PROJECT_DIR,'for_test')
    os.makedirs(path_to_folder_research,exist_ok= True)
    num_folder = ''
    if len(os.listdir(path_to_folder_research)):
        list_ = list(filter(lambda x: os.path.isdir(os.path.join(path_to_folder_research,x)), os.listdir(path_to_folder_research)))
        last = sorted(list_,key = lambda x:int(x.split('_')[-1]))[-1]
        num_folder = str(int(last.split('_')[-1])+1)
    else:
        num_folder = '0'
    os.makedirs(os.path.join(path_to_folder_research, name + '_' + num_folder))
    path_to_folder = os.path.join(path_to_folder_research, name + '_' + num_folder)
    return path_to_folder
def get_volume(path_to_dicom,path_to_model,path_to_final_folder):
    to_dicom = os.path.join(path_to_final_folder, 'DICOM')
    to_nifti = os.path.join(path_to_final_folder,'NIFTI')
    to_image = os.path.join(path_to_final_folder,'SLICES')
    to_predict = os.path.join(path_to_final_folder,'PREDICTED')
    to_mask = os.path.join(path_to_final_folder,'MASK')

    os.makedirs(to_dicom,exist_ok= True)
    os.makedirs(to_nifti,exist_ok= True)
    os.makedirs(to_image, exist_ok=True)
    os.makedirs(to_predict, exist_ok=True)
    os.makedirs(to_mask, exist_ok=True)


    #номер исследования
    num_research = path_to_dicom.split(os.path.sep)[-1]
    #Поиск и сохранение DICOM
    print("Текущее исследование: " + num_research)
    for root, dirs, files in os.walk(path_to_dicom):
        # search for DICOM
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                dicom_content = pydicom.dcmread(file_path)
            except InvalidDicomError:
                print("Невозможно прочитать файл: " + fname)
                continue
            try:
                orientation = np.round(dicom_content.ImageOrientationPatient)
            except AttributeError:
                continue
            # search for volume with contrast
            if 'FLAIR'.lower() in dicom_content.SeriesDescription.lower() and (
                    all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])):
                file_path_for_nifti = deepcopy(file_path)
                shutil.copy(file_path_for_nifti, to_dicom)

    #перевод DICOM в NIFTI
    dicom2nifti.convert_directory(to_dicom,to_nifti, compression=True, reorient=True)

    #сохранение изображений - срезов
    try:
        _, data_nifti = marked_mri_processing.get_nifti_meta(to_nifti)
    except TypeError:
        create_csv(path_to_final_folder, num_research,True)
        return

    for i in range(data_nifti.shape[2]):
        plt.imshow(data_nifti[:, :, i], cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(os.path.join(to_image,f"slice_{i}.png"), bbox_inches='tight', pad_inches=0)

    for path_to_image in glob.glob(os.path.join(to_image,'*.png')):
        image_, mask = yolo_train.example_case(path_to_image=path_to_image, path_to_model=path_to_model, save_image=True)
        plt.imshow(image_, cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(os.path.join(to_predict, path_to_image.split(os.path.sep)[-1]), bbox_inches='tight', pad_inches=0)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(os.path.join(to_mask, path_to_image.split(os.path.sep)[-1]), bbox_inches='tight', pad_inches=0)

    create_csv(path_to_final_folder,num_research)


def create_csv(path_to_nifti,number_research,error_ = False):
    # Путь к папке с обработанными данными
    processed_data_folder = path_to_nifti
    global_path_for_file = os.path.sep.join(processed_data_folder.split(os.path.sep)[:-1])
    # Создаем CSV файл для записи
    csv_file_path = os.path.join(global_path_for_file, 'brain_volumes.csv')

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        fields = ["Номер мозга","Объем белых областей (в мм^3)"]
        if os.stat(csv_file_path).st_size == 0:
            writer.writerow(fields)
        # Итерация по всем папкам внутри папки processed_data
        if error_:
            writer.writerow([number_research, "Неверный формат данных"])
        else:
            for root, dirs, files in os.walk(processed_data_folder):
                # Проверяем, есть ли папка NIFTI в текущем каталоге
                if 'NIFTI' in dirs:
                    # Формируем путь к папке NIFTI
                    nifti_folder = os.path.join(root, 'NIFTI')
                    # Получаем список всех файлов в папке NIFTI
                    nifti_files = os.listdir(nifti_folder)
                    # Итерация по всем файлам в папке NIFTI
                    for nifti_file in nifti_files:
                        # Проверяем, что файл имеет расширение .nii.gz
                        if nifti_file.endswith('.nii.gz'):
                            # Загружаем файл NIfTI
                            nifti_file_path = os.path.join(nifti_folder, nifti_file)
                            nifti_img = nib.load(nifti_file_path)
                            # Получаем размеры пикселей в мм
                            pixel_spacing_mm = nifti_img.header.get_zooms()
                            # Вывод информации о размерах пикселей
                            print("Pixel spacing (в мм) для файла", nifti_file, ":", pixel_spacing_mm)
                            # Формируем путь к папке annotation внутри папки Yolo
                            annotation_folder = os.path.join(root, 'MASK')
                            # Получаем список всех файлов в папке annotation
                            mask_files = os.listdir(annotation_folder)
                            # Инициализация списка для хранения объема белых областей на каждом слое
                            white_area_volumes = []
                            # Итерация по всем файлам в папке annotation
                            for idx, mask_file in enumerate(mask_files):
                                # Проверяем, что файл - изображение в формате PNG
                                if mask_file.endswith('.png'):
                                    # Загружаем маску
                                    mask_path = os.path.join(annotation_folder, mask_file)
                                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                    # Подсчитываем белые пиксели на маске
                                    white_pixels = cv2.countNonZero(mask)
                                    # Подсчитываем площадь белых областей на маске
                                    white_area_mm2 = white_pixels * pixel_spacing_mm[0] * pixel_spacing_mm[1]
                                    # Добавляем объем белых областей в список
                                    white_area_volumes.append(white_area_mm2)
                                    # Отображаем номер слоя и объем белых областей на этом слое
                                    # print("Слой:", idx + 1, "Объем белых областей (в мм^3):", white_area_mm2)
                            # Вывод информации о площади белых областей
                            print("Объем белых областей на каждом слое (в мм^3) для файла", nifti_file, ":",
                                  sum(white_area_volumes))
                            # Записываем информацию в CSV файл
                            writer.writerow([number_research, sum(white_area_volumes)])
    # Выводим путь к созданному CSV файлу
    #print("CSV файл успешно создан:", csv_file_path)

def all_research(path_to_all_research):
    for folder_ in os.listdir(path_to_all_research):
        path_to_one_research = os.path.join(path_to_all_research,folder_)
        get_volume(path_to_one_research,os.path.join(path_routing.PROJECT_DIR,'последние_метрики_26.03.24/weights/best.pt'),folder_creator())

all_research(r'C:\Users\Нинтендо\Desktop\1') #сюда  путь до исследований