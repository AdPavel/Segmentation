import datetime
from functools import partialmethod
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchio as tio
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm.auto import tqdm, trange
import os
import glob
import matplotlib.cm as cm
import matplotlib.animation as animation
from PIL import Image


figsize = 16, 9
class ColorMapHelper:
    def __init__(self, colors_path):
        columns = 'name', *'rgba'
        df = pd.read_csv(colors_path, sep=' ', header=None, index_col=0, names=columns)
        max_index = max(df.index)
        self.cmap_data = np.zeros((max_index + 1, 4))
        df[['r', 'g', 'b', 'a']] = df[['r', 'g', 'b', 'a']] / 255
        self.cmap_data[df.index] = df[['r', 'g', 'b', 'a']].values
        self.cmap_data[:, 3] = 1
        self.cmap = ListedColormap(self.cmap_data)

#функция считает атлас для T1 снимков
def create_atlas(path_to_nifti_mri):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    tio.Subject.plot = partialmethod(tio.Subject.plot, reorient=False)

    mri_path2_data = nib.load(path_to_nifti_mri).get_fdata()
    #print(mri_path2_data.shape)
    subject_oasis = tio.Subject(t1=tio.ScalarImage(path_to_nifti_mri))
    subject = subject_oasis
    #print(subject.shape)

    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.Crop((0, 0, 30, 30, 0, 0)),
    ]
    transform = tio.Compose(transforms)
    preprocessed = transform(subject)
    #subject.plot()
    #preprocessed.plot()
    repo = 'fepegar/highresnet'
    model_name = 'highres3dnet'
    model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model.to(device).eval()

    helper = ColorMapHelper(r'/GIFNiftyNet.ctbl')
    cmap = helper.cmap
    cmap_dict = {}

    input_tensor = preprocessed.t1.data[np.newaxis].to(device)  # add batch dim
    with torch.autocast(device.type):
        logits = model(input_tensor)#.to(device)
    full_volume_output_tensor = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True).cpu()[0]  # get first along batch dim
    seg = tio.LabelMap(tensor=full_volume_output_tensor, affine=preprocessed.t1.affine)
    name = 'parcellation'
    preprocessed.add_image(seg, name)
    cmap_dict[name] = cmap
    tensor_data = preprocessed['parcellation'].data.squeeze().numpy()#.plot() #cmap_dict=cmap_dict
    print(tensor_data.shape)
    output_as_gif(data_nifti=tensor_data,cmap = cmap)
    #get_same_t2(r'C:\Segmentation\Segmentation\database\processed_data\689\NIFTI\5_t2_flair_tra_fs_brain.nii.gz',cmap)
    # for i in range(tensor_data.shape[-1]):
    #     plt.figure(figsize=figsize)
    #     plt.imshow(tensor_data[..., i],cmap = cmap)
    #     plt.title(f"Segmented Slice {i}")
    #     plt.axis('off')
    #     plt.show(block = False)#
    #     plt.pause(.5)
    #     plt.close()

#Ищет срезы на T1 которые совпадут с T2
def get_same_t2(path_to_t2,cmap = cm.Greys_r):
    tio.Subject.plot = partialmethod(tio.Subject.plot, reorient=False)

    mri_path2_data = nib.load(path_to_t2).get_fdata()
    print(mri_path2_data.shape)
    subject_oasis = tio.Subject(t2=tio.ScalarImage(path_to_t2))
    subject = subject_oasis
    print(subject.shape)

    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.Crop((0, 0, 30, 30, 0, 0)),
    ]
    transform = tio.Compose(transforms)
    preprocessed = transform(subject)
    #subject.plot()
    #preprocessed.plot()

    tensor_data = preprocessed['t2'].data.squeeze().numpy()
    print(tensor_data.shape)
    output_as_gif(data_nifti=tensor_data,cmap=cmap)
    # for i in range(tensor_data.shape[-1]):
    #     plt.figure(figsize=figsize)
    #     plt.imshow(tensor_data[..., i], cmap=cmap)
    #     plt.title(f"Segmented Slice {i}")
    #     plt.axis('off')
    #     plt.show(block = False)#
    #     plt.pause(.1)
    #     plt.close()
    #     plt.show()


#функция выводит гифкой MRI снимки
def output_as_gif(path_to_Image_nifti = None,data_nifti = None,data_from_file_im = None,cmap = cm.Greys_r):
    if path_to_Image_nifti:
        epi_img = nib.load(path_to_Image_nifti).get_fdata()
    elif data_nifti:
        epi_img = data_nifti
    elif data_from_file_im:
        epi_img = []

        pp = sorted(os.listdir(data_from_file_im),key = lambda x: int(x.split('_')[1].split('.')[0]))
        for img in pp:
            path = os.path.join(data_from_file_im,img)
            image = Image.open(path)
            epi_img.append(image)
        ss = os.path.join(r'/for_test/research_0', 'output.gif')
        epi_img[0].save(ss,save_all = True,append_images = epi_img[1:],optimize=True,loop = 0,duration=200)
        return
    frames = []  # for storing the generated images
    fig = plt.figure(figsize=(16,9))
    for i in range(epi_img.shape[-1]):
        frames.append([plt.imshow(epi_img[...,i], cmap=cmap, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=1000)
    plt.axis('off')
    # ani.save('movie.mp4')
    plt.show()

path_to_needed = r'/for_test/research_0/PREDICTED'
#create_atlas(path_to_needed)
#get_same_t2(path_to_needed)

output_as_gif(data_from_file_im = path_to_needed)