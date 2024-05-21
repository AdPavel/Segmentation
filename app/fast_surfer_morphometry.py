import os
import shutil
import pydicom
import numpy as np
import dicom2nifti
from config.settings import path_routing
from copy import deepcopy
from pydicom.errors import InvalidDicomError


def get_nifti_t1_files(root_dir):
    # Создаем список для хранения абсолютных путей до файлов
    nifti_t1_files = []

    # Проходимся по всем папкам и файлам в корневом каталоге
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Проверяем, находится ли папка NIFTI_T1 в текущей директории
        if os.path.basename(dirpath) == 'NIFTI_T1':
            # Если да, то добавляем все файлы из этой папки в список
            for filename in filenames:
                nifti_t1_files.append(os.path.join(dirpath, filename))

    return nifti_t1_files


def fastsurfer(t1_image_paths):
    container_count = 0
    lic_data = path_routing.lic_path

    for t1_image_path in t1_image_paths:
        # Check if T1 image file exists
        if not os.path.isfile(t1_image_path):
            print("Error: T1 image file not found at", t1_image_path)
            continue

        # Generate unique SUBJECT ID based on filename
        filename = os.path.basename(t1_image_path)
        subject_id = os.path.splitext(filename)[0]
        # subjects_data = r'C:\Users\1\PycharmProjects\Segmentation\database\FS_my_mri_data'
        # Run the FastSurfer script
        if container_count < 2:
            print("Running the FastSurfer detached container for", subject_id)
            command = [
                # GPU version
                'docker run --gpus all',
                # mount :/data folder into a container
                '-v', t1_image_path, ':/data',
                # mount :/output folder into a container
                '-v', t1_image_path, ':/output',
                # mount :/fs_license folder into a container
                '-v', lic_data, ':/fs_license',
                # detached container
                '-d deepmi/fastsurfer:gpu-v2.1.2 --allow_root',
                # apply licence
                '--fs_license /fs_license/license.txt',
                # nii.gz from :/data
                '--t1 /data/',
                # device choice
                '--device cuda',
                # create output sub folder
                '--sid', subject_id,
                # output folder
                '--sd /output',
                # allow parallel calculating
                '--parallel'
            ]
            container_count = container_count + 1
        else:
            print("Running the FastSurfer container for", subject_id)
            command = [
                # GPU version
                'docker run --gpus all',
                # mount :/data folder into a container
                '-v', t1_image_path, ':/data',
                # mount :/output folder into a container
                '-v', t1_image_path, ':/output',
                # mount :/fs_license folder into a container
                '-v', lic_data, ':/fs_license',
                # container
                'deepmi/fastsurfer:gpu-v2.1.2 --allow_root',
                # apply licence
                '--fs_license /fs_license/license.txt',
                # nii.gz from :/data
                '--t1 /data/',
                # device choice
                '--device cuda',
                # create output sub folder
                '--sid', subject_id,
                # output folder
                '--sd /output',
                # allow parallel calculating
                '--parallel'
            ]
            container_count = 0


fastsurfer(get_nifti_t1_files(path_routing.morphometry_path))

