import os
import shutil
import pydicom
import numpy as np
import dicom2nifti
from copy import deepcopy

from pydicom.errors import InvalidDicomError


def process_dicom_folder(folder_path, to_dicom):
    for root, dirs, files in os.walk(folder_path):
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
            if 'FLAIR'.lower() in dicom_content.SeriesDescription.lower() and (
                    all(orientation[:3] == [1, 0, 0]) and all(orientation[3:] == [0, 1, 0])):
                shutil.copy(file_path, to_dicom)

path_to_dicom = "/путь/к/исходной/папке/DICOM"

# Определение пути к конечной папке DICOM
to_dicom = os.path.join(path_to_final_folder, 'DICOM')

# Переменная для хранения конечной папки с NIFTI-файлами
to_nifti = "/путь/к/конечной/папке/NIFTI"
t1_image_files = [os.path.join(to_nifti, file) for file in os.listdir(to_nifti) if os.path.isfile(os.path.join(to_nifti, file))]

# Переменная для хранения номера исследования
num_research = os.path.basename(path_to_dicom)

# Конвертация DICOM в NIFTI
dicom2nifti.convert_directory(to_dicom, to_nifti, compression=True, reorient=True)

def freesurfer(t1_image_paths):
    container_count = 0
    # Set FREESURFER_HOME environment variable
    # os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/7.4.1"

    # Append FREESURFER bin directory to PATH
    # os.environ["PATH"] += os.pathsep + os.path.join(os.environ["FREESURFER_HOME"], "bin")

    # Set SUBJECTS_DATA environment variable
    # subjects_data = '/home/dada/my_fastsurfer_analysis'
    # os.environ['SUBJECTS_DATA'] = subjects_data

    # Change directory to FastSurfer
    # fastsurfer_dir = '/home/dada/FastSurfer'
    # os.chdir(fastsurfer_dir)

    # Iterate through each T1 image file
    for t1_image_path in t1_image_paths:
        # Check if T1 image file exists
        if not os.path.isfile(t1_image_path):
            print("Error: T1 image file not found at", t1_image_path)
            continue

        # Generate unique SUBJECT ID based on filename
        filename = os.path.basename(t1_image_path)
        subject_id = os.path.splitext(filename)[0]
        subjects_data = r'C:\Users\1\PycharmProjects\Segmentation\database\FS_my_mri_data'
        lic_data = r'C:\Users\1\PycharmProjects\Segmentation\database\FS_my_fs_license_dir'
        # Run the FastSurfer script
        if container_count < 2:
            print("Running the FastSurfer detached container for", subject_id)
            command = [
                # GPU version
                'docker run --gpus all',
                # mount :/data folder into a container
                '-v', t1_image_path,':/data',
                # mount :/output folder into a container
                '-v', subjects_data, ':/output',
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
                '--sd /output', subjects_data,
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
                '-v', subjects_data, ':/output',
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
                '--sd /output', subjects_data,
                # allow parallel calculating
                '--parallel'
            ]
            container_count = 0
freesurfer(t1_image_files)

