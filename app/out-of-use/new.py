import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from skimage import color, transform, segmentation
from skimage.transform import resize
import cv2
from PIL import Image

img_path = r'E:\Segmentation\testy\wmparc.mgz'
img = nib.load(img_path)
data = img.get_fdata()

labels_rgb = [
    [0, 0, 0],               # 0 Background
    [0, 0, 255],             # 4 Non-ventricular
    [127, 255, 212],         # 5 3rd-Ventricle
    [240, 230, 140],         # 6 4th-Ventricle
    [176, 48, 96],           # 7 5th-Ventricle
    [48, 176, 96],           # 8 Right-Accumbens-Area
    [48, 176, 96],           # 9 Left-Accumbens-Area
    [103, 255, 255],         # 10 Right-Amygdala
    [103, 255, 255],         # 11 Left-Amygdala
    [238, 186, 243],         # 12 Pons
    [119, 159, 176],         # 13 Brain-Stem
    [122, 186, 220],         # 14 Right-Caudate
    [122, 186, 220],         # 15 Left-Caudate
    [96, 204, 96],           # 16 Right-Cerebellum-Exterior
    [96, 204, 96],           # 17 Left-Cerebellum-Exterior
    [220, 247, 164],         # 18 Right-Cerebellum
    [220, 247, 164],         # 19 Left-Cerebellum
    [60, 60, 60],            # 22 3rd-Ventricle-(Posterior-part)
    [220, 216, 20],          # 23 Right-Hippocampus
    [220, 216, 20],          # 24 Left-Hippocampus
    [196, 58, 250],          # 25 Right-Inf-Lat-Vent
    [196, 58, 250],          # 26 Left-Inf-Lat-Vent
    [120, 18, 134],          # 27 Right-Lateral-Ventricle
    [120, 18, 134],          # 28 Left-Lateral-Ventricle
    [12, 48, 255],           # 29 Right-Pallidum
    [12, 48, 225],           # 30 Left-Pallidum
    [236, 13, 176],          # 31 Right-Putamen
    [236, 13, 176],          # 32 Left-Putamen
    [0, 118, 14],            # 33 Right-Thalamus-Proper
    [0, 118, 14],            # 34 Left-Thalamus-Proper
    [165, 42, 42],           # 35 Right-Ventral-DC
    [165, 42, 42],           # 36 Left-Ventral-DC
    [160, 32, 240],          # 37 Right-vessel
    [160, 32, 240],          # 38 Left-vessel
    [56, 192, 255],          # 39 Right-periventricular-white-matter
    [56, 192, 255],          # 40 Left-periventricular-white-matter
    [255, 225, 225],         # 41 Optic-Chiasm
    [184, 237, 194],         # 42 Cerebellar-Vermal-Lobules-I-V
    [180, 231, 250],         # 43 Cerebellar-Vermal-Lobules-VI-VII
    [225, 183, 231],         # 44 Cerebellar-Vermal-Lobules-VIII-X
    [180, 180, 180],         # 45 Left-Basal-Forebrain
    [180, 180, 180],         # 46 Right-Basal-Forebrain
    [245, 255, 200],         # 47 Right-Temporal-White-Matter
    [255, 230, 255],         # 48 Right-Insula-White-Matter
    [245, 245, 245],         # 49 Right-Cingulate-White-Matter
    [220, 255, 220],         # 50 Right-Frontal-White-Matter
    [220, 220, 220],         # 51 Right-Occipital-White-Matter
    [200, 255, 255],         # 52 Right-Parietal-White-Matter
    [250, 220, 200],         # 53 Corpus-Callosum
    [245, 255, 200],         # 54 Left-Temporal-White-Matter
    [255, 230, 255],         # 55 Left-Insula-White-Matter
    [245, 245, 245],         # 56 Left-Cingulate-White-Matter
    [220, 255, 220],         # 57 Left-Frontal-White-Matter
    [220, 220, 220],         # 58 Left-Occipital-White-Matter
    [200, 255, 255],         # 59 Left-Parietal-White-Matter
    [140, 125, 255],         # 60 Right-Claustrum
    [140, 125, 255],         # 61 Left-Claustrum
    [255, 62, 150],          # 62 Right-ACgG-anterior-cingulate-gyrus
    [255, 62, 150],          # 63 Left-ACgG-anterior-cingulate-gyrus
    [160, 82, 45],           # 64 Right-AIns-anterior-insula
    [160, 82, 45],           # 65 Left-AIns-anterior-insula
    [165, 42, 42],           # 66 Right-AOrG-anterior-orbital-gyrus
    [165, 42, 42],           # 67 Left-AOrG-anterior-orbital-gyrus
    [205, 91, 69],           # 68 Right-AnG-angular-gyrus
    [205, 91, 69],           # 69 Left-AnG-angular-gyrus
    [100, 149, 237],         # 70 Right-Calc-calcarine-cortex
    [100, 149, 237],         # 71 Left-Calc-calcarine-cortex
    [135, 206, 235],         # 72 Right-CO-central-operculum
    [135, 206, 235],         # 73 Left-CO-central-operculum
    [250, 128, 114],         # 74 Right-Cun-cuneus
    [250, 128, 114],         # 75 Left-Cun-cuneus
    [255, 255, 0],           # 76 Right-Ent-entorhinal-area
    [255, 255, 0],           # 77 Left-Ent-entorhinal-area
    [221, 160, 221],         # 78 Right-FO-frontal-operculum
    [221, 160, 221],         # 79 Left-FO-frontal-operculum
    [0, 238, 0],             # 80 Right-FRP-frontal-pole
    [0, 238, 0],             # 81 Left-FRP-frontal-pole
    [205, 92, 92],           # 82 Right-FuG-fusiform-gyrus
    [205, 92, 92],           # 83 Left-FuG-fusiform-gyrus
    [176, 48, 96],           # 84 Right-GRe-gyrus-rectus
    [176, 48, 96],           # 85 Left-GRe-gyrus-rectus
    [152, 251, 152],         # 86 Right-IOG-inferior-occipital-gyrus
    [152, 251, 152],         # 87 Left-IOG-inferior-occipital-gyrus
    [50, 205, 50],           # 88 Right-ITG-inferior-temporal-gyrus
    [50, 205, 50],           # 89 Left-ITG-inferior-temporal-gyrus
    [0, 100, 0],             # 90 Right-LiG-lingual-gyrus
    [0, 100, 0],             # 91 Left-LiG-lingual-gyrus
    [173, 216, 230],         # 92 Right-LOrG-lateral-orbital-gyrus
    [173, 216, 230],         # 93 Left-LOrG-lateral-orbital-gyrus
    [153, 50, 204],          # 94 Right-MCgG-middle-cingulate-gyrus
    [153, 50, 204],          # 95 Left-MCgG-middle-cingulate-gyrus
    [160, 32, 240],          # 96 Right-MFC-medial-frontal-cortex
    [160, 32, 240],          # 97 Left-MFC-medial-frontal-cortex
    [0, 206, 208],           # 98 Right-MFG-middle-frontal-gyrus
    [0, 206, 208],           # 99 Left-MFG-middle-frontal-gyrus
    [51, 50, 135],           # 100 Right-MOG-middle-occipital-gyrus
    [51, 50, 135],           # 101 Left-MOG-middle-occipital-gyrus
    [135, 50, 74],           # 102 Right-MOrG-medial-orbital-gyrus
    [135, 50, 74],           # 103 Left-MOrG-medial-orbital-gyrus
    [218, 112, 214],         # 104 Right-MPoG-postcentral-gyrus-medial-segment
    [218, 112, 214],         # 105 Left-MPoG-postcentral-gyrus-medial-segment
    [240, 230, 140],         # 106 Right-MPrG-precentral-gyrus-medial-segment
    [240, 230, 140],         # 107 Left-MPrG-precentral-gyrus-medial-segment
    [255, 255, 0],           # 108 Right-MSFG-superior-frontal-gyrus-medial-segment
    [255, 255, 0],           # 109 Left-MSFG-superior-frontal-gyrus-medial-segment
    [255, 110, 180],         # 110 Right-MTG-middle-temporal-gyrus
    [255, 110, 180],         # 111 Left-MTG-middle-temporal-gyrus
    [0, 255, 255],           # 112 Right-OCP-occipital-pole
    [0, 255, 255],           # 113 Left-OCP-occipital-pole
    [100, 50, 100],          # 114 Right-OFuG-occipital-fusiform-gyrus
    [100, 50, 100],          # 115 Left-OFuG-occipital-fusiform-gyrus
    [178, 34, 34],           # 116 Right-OpIFG-opercular-part-of-the-inferior-frontal-gyrus
    [178, 34, 34],           # 117 Left-OpIFG-opercular-part-of-the-inferior-frontal-gyrus
    [255, 0, 255],           # 118 Right-OrIFG-orbital-part-of-the-inferior-frontal-gyrus
    [255, 0, 255],           # 119 Left-OrIFG-orbital-part-of-the-inferior-frontal-gyrus
    [39, 64, 139],           # 120 Right-PCgG-posterior-cingulate-gyrus
    [39, 64, 139],           # 121 Left-PCgG-posterior-cingulate-gyrus
    [255, 99, 71],           # 122 Right-PCu-precuneus
    [255, 99, 71],           # 123 Left-PCu-precuneus
    [255, 69, 0],            # 124 Right-PHG-parahippocampal-gyrus
    [255, 69, 0],            # 125 Left-PHG-parahippocampal-gyrus
    [210, 180, 140],         # 126 Right-PIns-posterior-insula
    [210, 180, 140],         # 127 Left-PIns-posterior-insula
    [0, 255, 127],           # 128 Right-PO-parietal-operculum
    [0, 255, 127],           # 129 Left-PO-parietal-operculum
    [74, 155, 60],           # 130 Right-PoG-postcentral-gyrus
    [74, 155, 60],           # 131 Left-PoG-postcentral-gyrus
    [255, 215, 0],           # 132 Right-POrG-posterior-orbital-gyrus
    [255, 215, 0],           # 133 Left-POrG-posterior-orbital-gyrus
    [238, 0, 0],             # 134 Right-PP-planum-polare
    [238, 0, 0],             # 135 Left-PP-planum-polare
    [46, 139, 87],           # 136 Right-PrG-precentral-gyrus
    [46, 139, 87],           # 137 Left-PrG-precentral-gyrus
    [238, 201, 0],           # 138 Right-PT-planum-temporale
    [238, 201, 0],           # 139 Left-PT-planum-temporale
    [102, 205, 170],         # 140 Right-SCA-subcallosal-area
    [102, 205, 170],         # 141 Left-SCA-subcallosal-area
    [255, 218, 185],         # 142 Right-SFG-superior-frontal-gyrus
    [255, 218, 185],         # 143 Left-SFG-superior-frontal-gyrus
    [238, 130, 238],         # 144 Right-SMC-supplementary-motor-cortex
    [238, 130, 238],         # 145 Left-SMC-supplementary-motor-cortex
    [255, 165, 0],           # 146 Right-SMG-supramarginal-gyrus
    [255, 165, 0],           # 147 Left-SMG-supramarginal-gyrus
    [255, 192, 203],         # 148 Right-SOG-superior-occipital-gyrus
    [255, 192, 203],         # 149 Left-SOG-superior-occipital-gyrus
    [244, 222, 179],         # 150 Right-SPL-superior-parietal-lobule
    [244, 222, 179],         # 151 Left-SPL-superior-parietal-lobule
    [208, 32, 144],          # 152 Right-STG-superior-temporal-gyrus
    [208, 32, 144],          # 153 Left-STG-superior-temporal-gyrus
    [34, 139, 34],           # 154 Right-TMP-temporal-pole
    [34, 139, 34],           # 155 Left-TMP-temporal-pole
    [125, 255, 212],         # 156 Right-TrIFG-triangular-part-of-the-inferior-frontal-gyrus
    [127, 255, 212],         # 157 Left-TrIFG-triangular-part-of-the-inferior-frontal-gyrus
    [0, 0, 128],             # 158 Right-TTG-transverse-temporal-gyrus
    [0, 0, 128]              # 159 Left-TTG-transverse-temporal-gyrus
]

labels = [
    "Background",
    "Non-ventricular",
    "3rd-Ventricle",
    "4th-Ventricle",
    "5th-Ventricle",
    "Right-Accumbens-Area",
    "Left-Accumbens-Area",
    "Right-Amygdala",
    "Left-Amygdala",
    "Pons",
    "Brain-Stem",
    "Right-Caudate",
    "Left-Caudate",
    "Right-Cerebellum-Exterior",
    "Left-Cerebellum-Exterior",
    "Right-Cerebellum",
    "Left-Cerebellum",
    "3rd-Ventricle-(Posterior-part)",
    "Right-Hippocampus",
    "Left-Hippocampus",
    "Right-Inf-Lat-Vent",
    "Left-Inf-Lat-Vent",
    "Right-Lateral-Ventricle",
    "Left-Lateral-Ventricle",
    "Right-Pallidum",
    "Left-Pallidum",
    "Right-Putamen",
    "Left-Putamen",
    "Right-Thalamus-Proper",
    "Left-Thalamus-Proper",
    "Right-Ventral-DC",
    "Left-Ventral-DC",
    "Right-vessel",
    "Left-vessel",
    "Right-periventricular-white-matter",
    "Left-periventricular-white-matter",
    "Optic-Chiasm",
    "Cerebellar-Vermal-Lobules-I-V",
    "Cerebellar-Vermal-Lobules-VI-VII",
    "Cerebellar-Vermal-Lobules-VIII-X",
    "Left-Basal-Forebrain",
    "Right-Basal-Forebrain",
    "Right-Temporal-White-Matter",
    "Right-Insula-White-Matter",
    "Right-Cingulate-White-Matter",
    "Right-Frontal-White-Matter",
    "Right-Occipital-White-Matter",
    "Right-Parietal-White-Matter",
    "Corpus-Callosum",
    "Left-Temporal-White-Matter",
    "Left-Insula-White-Matter",
    "Left-Cingulate-White-Matter",
    "Left-Frontal-White-Matter",
    "Left-Occipital-White-Matter",
    "Left-Parietal-White-Matter",
    "Right-Claustrum",
    "Left-Claustrum",
    "Right-ACgG-anterior-cingulate-gyrus",
    "Left-ACgG-anterior-cingulate-gyrus",
    "Right-AIns-anterior-insula",
    "Left-AIns-anterior-insula",
    "Right-AOrG-anterior-orbital-gyrus",
    "Left-AOrG-anterior-orbital-gyrus",
    "Right-AnG-angular-gyrus",
    "Left-AnG-angular-gyrus",
    "Right-Calc-calcarine-cortex",
    "Left-Calc-calcarine-cortex",
    "Right-CO-central-operculum",
    "Left-CO-central-operculum",
    "Right-Cun-cuneus",
    "Left-Cun-cuneus",
    "Right-Ent-entorhinal-area",
    "Left-Ent-entorhinal-area",
    "Right-FO-frontal-operculum",
    "Left-FO-frontal-operculum",
    "Right-FRP-frontal-pole",
    "Left-FRP-frontal-pole",
    "Right-FuG-fusiform-gyrus",
    "Left-FuG-fusiform-gyrus",
    "Right-GRe-gyrus-rectus",
    "Left-GRe-gyrus-rectus",
    "Right-IOG-inferior-occipital-gyrus",
    "Left-IOG-inferior-occipital-gyrus",
    "Right-ITG-inferior-temporal-gyrus",
    "Left-ITG-inferior-temporal-gyrus",
    "Right-LiG-lingual-gyrus",
    "Left-LiG-lingual-gyrus",
    "Right-LOrG-lateral-orbital-gyrus",
    "Left-LOrG-lateral-orbital-gyrus",
    "Right-MCgG-middle-cingulate-gyrus",
    "Left-MCgG-middle-cingulate-gyrus",
    "Right-MFC-medial-frontal-cortex",
    "Left-MFC-medial-frontal-cortex",
    "Right-MFG-middle-frontal-gyrus",
    "Left-MFG-middle-frontal-gyrus",
    "Right-MOG-middle-occipital-gyrus",
    "Left-MOG-middle-occipital-gyrus",
    "Right-MOrG-medial-orbital-gyrus",
    "Left-MOrG-medial-orbital-gyrus",
    "Right-MPoG-postcentral-gyrus-medial-segment",
    "Left-MPoG-postcentral-gyrus-medial-segment",
    "Right-MPrG-precentral-gyrus-medial-segment",
    "Left-MPrG-precentral-gyrus-medial-segment",
    "Right-MSFG-superior-frontal-gyrus-medial-segment",
    "Left-MSFG-superior-frontal-gyrus-medial-segment",
    "Right-MTG-middle-temporal-gyrus",
    "Left-MTG-middle-temporal-gyrus",
    "Right-OCP-occipital-pole",
    "Left-OCP-occipital-pole",
    "Right-OFuG-occipital-fusiform-gyrus",
    "Left-OFuG-occipital-fusiform-gyrus",
    "Right-OpIFG-opercular-part-of-the-inferior-frontal-gyrus",
    "Left-OpIFG-opercular-part-of-the-inferior-frontal-gyrus",
    "Right-OrIFG-orbital-part-of-the-inferior-frontal-gyrus",
    "Left-OrIFG-orbital-part-of-the-inferior-frontal-gyrus",
    "Right-PCgG-posterior-cingulate-gyrus",
    "Left-PCgG-posterior-cingulate-gyrus",
    "Right-PCu-precuneus",
    "Left-PCu-precuneus",
    "Right-PHG-parahippocampal-gyrus",
    "Left-PHG-parahippocampal-gyrus",
    "Right-PIns-posterior-insula",
    "Left-PIns-posterior-insula",
    "Right-PO-parietal-operculum",
    "Left-PO-parietal-operculum",
    "Right-PoG-postcentral-gyrus",
    "Left-PoG-postcentral-gyrus",
    "Right-POrG-posterior-orbital-gyrus",
    "Left-POrG-posterior-orbital-gyrus",
    "Right-PP-planum-polare",
    "Left-PP-planum-polare",
    "Right-PrG-precentral-gyrus",
    "Left-PrG-precentral-gyrus",
    "Right-PT-planum-temporale",
    "Left-PT-planum-temporale",
    "Right-SCA-subcallosal-area",
    "Left-SCA-subcallosal-area",
    "Right-SFG-superior-frontal-gyrus",
    "Left-SFG-superior-frontal-gyrus",
    "Right-SMC-supplementary-motor-cortex",
    "Left-SMC-supplementary-motor-cortex",
    "Right-SMG-supramarginal-gyrus",
    "Left-SMG-supramarginal-gyrus",
    "Right-SOG-superior-occipital-gyrus",
    "Left-SOG-superior-occipital-gyrus",
    "Right-SPL-superior-parietal-lobule",
    "Left-SPL-superior-parietal-lobule",
    "Right-STG-superior-temporal-gyrus",
    "Left-STG-superior-temporal-gyrus",
    "Right-TMP-temporal-pole",
    "Left-TMP-temporal-pole",
    "Right-TrIFG-triangular-part-of-the-inferior-frontal-gyrus",
    "Left-TrIFG-triangular-part-of-the-inferior-frontal-gyrus",
    "Right-TTG-transverse-temporal-gyrus",
    "Left-TTG-transverse-temporal"
]

labels_rgb_tuples = [tuple(rgb) for rgb in labels_rgb]
dictionary = dict(zip(labels_rgb_tuples, labels))
dictionary = dict(zip(labels, labels_rgb))
print(dictionary)

cmap = ListedColormap(np.array(labels_rgb) / 255.0)

pred_slice = data[:, 90, :]

color_grid = color.label2rgb(pred_slice, bg_label=0, colors=cmap.colors)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))

axes[0].imshow(color_grid, interpolation='nearest')
axes[0].set_title('Prediction with Labeled Colors')


png_image_1 = plt.imread(r'E:\Segmentation\for_test\research_16\PREDICTED\slice_56.png')
png_image_resized_1 = transform.resize(png_image_1, color_grid.shape[:2])

axes[1].imshow(png_image_resized_1, interpolation='nearest')
axes[1].set_title('Overlay Image 1')

png_image_2 = plt.imread(r'E:\Segmentation\for_test\research_16\PREDICTED\slice_56.png')
png_image_resized_2 = transform.resize(png_image_2, color_grid.shape[:2])

axes[1].imshow(png_image_resized_2, interpolation='nearest', alpha=0.5)

plt.show()

gray_image_2 = cv2.cvtColor(png_image_2, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(gray_image_2, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.zeros_like(png_image_2)
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
contour_image_resized = transform.resize(contour_image, color_grid.shape[:2])

axes[1].imshow(contour_image_resized, interpolation='nearest', alpha=0.5)

plt.show()

alpha = 0.5  # Set alpha to 0.1 for color_grid dominance
overlay_image = (1 - alpha) * png_image_resized_2[:, :, :3] + alpha * color_grid

# Create new figure and axes for displaying the overlay
fig, axes = plt.subplots(figsize=(10, 5))

axes.imshow(overlay_image, interpolation='nearest')
axes.set_title('Overlay')

plt.show()