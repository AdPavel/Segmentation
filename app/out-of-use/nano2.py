import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Загрузка первого и второго NIfTI файлов
nifti_file_1 = r'C:\Segmentation\T\5_t2_flair_tra_fs_brain.nii.gz'
nifti_file_2 = r'C:\Segmentation\T\11_t1_tra_ce.nii.gz'

# Загрузка данных из NIfTI-файлов
try:
    nifti_data_1 = nib.load(nifti_file_1)
    nifti_data_2 = nib.load(nifti_file_2)
except FileNotFoundError:
    print("Ошибка: Файл не найден.")
except nib.filebasedimages.ImageFileError:
    print("Ошибка: Некорректный формат файла.")

# Аффинная матрица для преобразования координат
affine_matrix = nifti_data_1.affine
affine_matrix2 = nifti_data_2.affine

# Получение информации о количестве срезов и их высоте
num_slices_1 = nifti_data_1.shape[-1]
num_slices_2 = nifti_data_2.shape[-1]

# Вывод информации о количестве срезов
print(f"В первом файле {num_slices_1} срезов.")
print(f"Второй файл содержит {num_slices_2} срезов.")

# Списки для хранения высот срезов и соответствующих номеров срезов
heights_1 = []
heights_2 = []
slice_numbers_1 = []
slice_numbers_2 = []

# Получение высоты каждого среза по оси Z для первого файла
for i in range(num_slices_1):
    slice_coords = np.array([0, 0, i, 1])
    z_height = np.dot(affine_matrix, slice_coords)[:3][2]
    heights_1.append(z_height)
    slice_numbers_1.append(i)

# Получение высоты каждого среза по оси Z для второго файла
for i in range(num_slices_2):
    slice_coords = np.array([0, 0, i, 1])
    z_height = np.dot(affine_matrix2, slice_coords)[:3][2]
    heights_2.append(z_height)
    slice_numbers_2.append(i)

# Находим и выводим одинаковые высоты срезов
common_heights = sorted(set(int(height) for height in heights_1) & set(int(height) for height in heights_2))
print("Одинаковые высоты срезов:")
for height in common_heights:
    # Получаем индексы срезов с одинаковой высотой
    indices_1 = [i for i, h in enumerate(heights_1) if int(h) == height]
    indices_2 = [i for i, h in enumerate(heights_2) if int(h) == height]

    # Выводим номера срезов
    for idx_1, idx_2 in zip(indices_1, indices_2):
        print(f"Высота {height} мм: Срез {slice_numbers_1[idx_1]} из файла 1, Срез {slice_numbers_2[idx_2]} из файла 2")

        # Отображение срезов на одном изображении
        slice_1 = nifti_data_1.get_fdata()[:, :, slice_numbers_1[idx_1]]
        slice_2 = nifti_data_2.get_fdata()[:, :, slice_numbers_2[idx_2]]

        # Отображение срезов из первого и второго файла
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(slice_1, cmap='gray')
        plt.title(f'Срез {slice_numbers_1[idx_1]} из файла 1')

        plt.subplot(1, 3, 2)
        plt.imshow(slice_2, cmap='gray')
        plt.title(f'Срез {slice_numbers_2[idx_2]} из файла 2')

        # Наложение срезов друг на друга
        plt.subplot(1, 3, 3)
        plt.imshow(slice_1, cmap='gray')
        plt.imshow(slice_2, cmap='jet', alpha=0.3)
        plt.title('Наложение срезов')

        plt.show()