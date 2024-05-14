import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.abspath("Segmentation"), '..','..'))
print(PROJECT_DIR)
MARKED_MRI_PATH = PROJECT_DIR + r"\database\marked_mri"
print(MARKED_MRI_PATH)
PROCESSED_DATA_PATH = PROJECT_DIR + r"\database\processed_data"
print(PROCESSED_DATA_PATH)
TRAIN_SET_PATH = PROJECT_DIR + r"\database\train_set"
print(TRAIN_SET_PATH)