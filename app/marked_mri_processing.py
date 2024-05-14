import json
import os
import shutil
from copy import deepcopy

import cv2
import dicom2nifti  # to convert DICOM files to the NIftI format
import matplotlib.pyplot as plt
import nibabel as nib  # nibabel to handle nifti files
import numpy as np
import pydicom  # pydicom to handle dicom files
import glob
import path_routing


def create_nifti_from_contrast_volumes(root_path, postpr_path):
    # walk the path and creates nifti from dicom in each survey then returns list of survey paths and list of paths to DICOM in marked_mri
    survey_list = []
    raw_data_list = []
    survey_name = ''
    file_path_for_nifti = ''
    survey_path = ''
    for root, dirs, files in os.walk(root_path):
        # search for DICOM
        for fname in files:

            if fname.startswith('IM'):
                file_path = os.path.join(root, fname)
                dicom_content = pydicom.dcmread(file_path)
                orientation = np.round(dicom_content.ImageOrientationPatient)

                # search for volume with contrast
                if 'FLAIR' in dicom_content.SeriesDescription and (
                        all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])):
                    file_path_for_nifti = deepcopy(file_path)

                    while not file_path.endswith('DICOM'):
                        file_path = os.path.abspath(os.path.join(os.path.abspath(file_path), '..'))
                    file_path = os.path.abspath(os.path.join(os.path.abspath(file_path), '..'))
                    survey_name = os.path.basename(file_path)

                    # make directories in processed data
                    raw_data_path = os.path.abspath(os.path.join(os.path.abspath(file_path)))
                    if not file_path in raw_data_list:
                        raw_data_list.append(os.path.abspath(file_path))
                    os.makedirs(f'{postpr_path}{os.path.sep}{survey_name}/DICOM', exist_ok=True)

                    os.makedirs(f'{postpr_path}{os.path.sep}{survey_name}/NIFTI', exist_ok=True)
                    shutil.copy(file_path_for_nifti, f'{postpr_path}{os.path.sep}{survey_name}/DICOM')
                    survey_path = os.path.abspath(f'{postpr_path}{os.path.sep}{survey_name}')
                    # add survey path to list
                    if not survey_path in survey_list:
                        survey_list.append(survey_path)
                    # create NIFTI dir
                    # os.makedirs(survey_path + '/NIFTI', exist_ok=True)
                    # create a nifti from this volume
        # print(os.path.dirname(os.path.abspath(file_path)))  # debug
        print(survey_name)
        print(survey_path)  # debug

        dicom2nifti.convert_directory(f'{postpr_path}{os.path.sep}{survey_name}/DICOM',
                                      survey_path + '/NIFTI', compression=True, reorient=True)

    return survey_list, raw_data_list


def get_nifti_meta(nifti_path):
    # create two variables from one nifti meta
    nifti_path = os.path.join(nifti_path, 'NIFTI') if nifti_path.split(os.path.sep)[-1] != 'NIFTI' else nifti_path
    file_path = ''
    for root, dirs, files in os.walk(nifti_path):
        for fname in files:
            file_path = os.path.join(root, fname)
    try:
        epi_img = nib.load(os.path.join(nifti_path, os.listdir(nifti_path)[0]))
    except IndexError:
        print('Отсутствуют файлы DICOM')
        return

    epi_img_data = epi_img.get_fdata()
    return epi_img, epi_img_data


def set_needed_file(postpr_path, root_path):
    for root, dirs, files in os.walk(postpr_path):
        if 'NIFTI' in dirs and root.split(os.path.sep)[-1] in os.listdir(root_path):
            os.makedirs(os.path.join(root, 'NIFTI_T1'), exist_ok=True)
            os.makedirs(os.path.join(root, 'DICOM_T1'), exist_ok=True)
            folder_ = root.split(f'{os.path.sep}')[-1]
            if folder_ in os.listdir(root_path):
                path = os.path.join(root_path, folder_, 'DICOM', '**')
                for file_ in glob.glob(path, recursive=True):
                    if file_.split(os.path.sep)[-1].startswith('IM'):
                        dicom_content = pydicom.dcmread(file_)
                        orientation = np.round(dicom_content.ImageOrientationPatient)  # ориентация
                        tra_in = 'tra' in dicom_content.SeriesDescription.lower()
                        t1_in = 'T1'.lower() in dicom_content.SeriesDescription.lower()
                        if t1_in and ((all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])) or tra_in):
                            path = os.path.join(postpr_path + os.path.sep + folder_, 'DICOM_T1',
                                                file_.split(os.path.sep)[-1])
                            shutil.copy(file_, path)
            print(folder_)
            dicom2nifti.convert_directory(os.path.join(postpr_path, folder_, 'DICOM_T1'),
                                          os.path.join(postpr_path, folder_, 'NIFTI_T1'), compression=True,
                                          reorient=True)


def create_demyelination_areas_markups(epi_img, epi_img_data, markup_path):
    # read jsons and create a list of matrixes with coordinates for one nifti
    points_lps = []
    markup_path += '/Markups'
    # walk through .json filenames and add their location to list
    for root, dirs, files in os.walk(markup_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as f:
                    local_point_list = []
                    json_data = json.load(f)
                    # form a 4x1 np matrix using 'orientation' and 'position' from each .json
                    for point in json_data["markups"][0]["controlPoints"]:
                        orientation_matrix = np.reshape(np.array(point['orientation']), (3, 3))
                        position_matrix = np.reshape(np.array(point["position"]), (3, 1))
                        local_point_list.append(
                            np.vstack(((orientation_matrix @ position_matrix), [1])))
                    points_lps.append(local_point_list)
    new_points_lps = []
    for point in points_lps:
        local_list = []
        for local_point in point:
            index_coords = np.round(np.dot(np.linalg.inv(epi_img.affine), np.array(local_point)))
            # index_coords = np.divide((np.array(local_point) - abc),epi_img.header.get_zooms())
            local_list.append(index_coords)
        new_points_lps.append(local_list)
    return new_points_lps


def create_yolo_data(epi_img, epi_img_data, new_points_lps, yolo_path):
    # create .txt for yolo
    mask_image_all = []  # Трехмерное хранилище масок
    os.makedirs(yolo_path + '/Yolo', exist_ok=True)
    os.makedirs(yolo_path + '/Yolo/annotation/', exist_ok=True)
    os.makedirs(yolo_path + '/Yolo/images/', exist_ok=True)
    path_to_Yolo_data = yolo_path + '/Yolo/'

    for i in range(epi_img.shape[2]):
        mask_image = np.zeros((epi_img.shape[0], epi_img.shape[1]))  # Содержи маску для одного среза

        # Создание контуров вокруг областей внутри точек
        for points in new_points_lps:
            if len(points) > 0 and int(points[0][2]) == i:
                # Выборка из X и Y для зополнения обастей
                contour = [[point[1][0], point[0][0]] for point in points]
                # Заполнение области внутри точек
                cv2.fillConvexPoly(mask_image, np.array(contour, 'int32'), 255)
        mask_image_all.append(mask_image)
    # for i,slice_ in enumerate(mask_image_all):
    #   plt.imshow(slice_, cmap=plt.cm.gray)
    #   plt.title(f"Слайс на глубине Z={i}")
    #   plt.show()
    some_doubts = np.transpose(mask_image_all, (1, 2, 0))
    for i in range(some_doubts.shape[2]):
        plt.imshow(some_doubts[:, :, i], cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(os.path.join(path_to_Yolo_data + 'annotation/', f"slice_{i}.png"), bbox_inches='tight',
                    pad_inches=0)
        plt.imshow(epi_img_data[:, :, i], cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(os.path.join(path_to_Yolo_data + 'images/', f"slice_{i}.png"), bbox_inches='tight', pad_inches=0)


def mask_to_polygons(path_to_mask):
    path_to_mask += '/Yolo/'
    os.makedirs(path_to_mask + 'labels/', exist_ok=True)
    os.makedirs(path_to_mask + '/labels/train', exist_ok=True)
    os.makedirs(path_to_mask + '/labels/val', exist_ok=True)

    input_dir = path_to_mask + 'annotation/'
    output_dir = path_to_mask + 'labels/'

    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        if j.endswith('png'):
            # load the binary mask and get its contours
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            H, W = mask.shape
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # convert the contours to polygons
            polygons = []
            for cnt in contours:
                # if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

            # print the polygons
            with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
                for polygon in polygons:
                    for p_, p in enumerate(polygon):
                        if p_ == len(polygon) - 1:
                            f.write('{}\n'.format(p))
                        elif p_ == 0:
                            f.write('0 {} '.format(p))
                        else:
                            f.write('{} '.format(p))

                f.close()


def raw_data_processing():
    survey_list, raw_data_list = create_nifti_from_contrast_volumes(root_path=path_routing.MARKED_MRI_PATH,
                                                                    postpr_path=path_routing.PROCESSED_DATA_PATH)
    for survey_path, raw_data_path in zip(survey_list, raw_data_list):
        epi_img, epi_img_data = get_nifti_meta(nifti_path=survey_path)
        new_points_lps = create_demyelination_areas_markups(epi_img, epi_img_data, markup_path=raw_data_path)
        create_yolo_data(epi_img, epi_img_data, new_points_lps, yolo_path=survey_path)
        mask_to_polygons(path_to_mask=survey_path)


#set_needed_file(path_routing.PROCESSED_DATA_PATH, path_routing.MARKED_MRI_PATH)
