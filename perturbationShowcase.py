import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg',force=True)

from datautils import process
import pandas as pd
import numpy as np
import tensorflow as tf
import os

fig,a =  plt.subplots(3,5)
fig.tight_layout(pad=.20)

fig.suptitle('Perturbation', fontsize=16)
fig.subplots_adjust(top=0.88)
for i in range(3):
    for j in range(5):
        plt.setp(a[i][j].get_xticklabels(), visible=False)
        plt.setp(a[i][j].get_yticklabels(), visible=False)
        a[i][j].set_xticks([])
        a[i][j].set_yticks([])
        a[i][j].tick_params(axis='both', which='both', length=0)
os.chdir(r"C:\Users\17033\Desktop\p2OPHAIresults\modelResults")
csv = r"C:\Users\17033\Desktop\p2OPHAIresults\RIGA_c\data_FullFundus_256_joint_orig_test.csv"
df_test = pd.read_csv(csv)[['imageID', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()

#Severity 3
""" 
RIGA_BinRushed1_0
"""
imgpath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_0.png"
model = tf.keras.models.load_model(r"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\unet.h5")
for i in df_test:
    if i[0] == "RIGA_BinRushed1_0.png":
        img, Fovea = process(imgpath, int(i[1]), int(i[2]), 256)
        img, Disk = process(imgpath, int(i[3]), int(i[4]), 256)
        a[0][0].imshow(img)
        a[0][0].set_title('Orig. Image', fontsize = 10)

        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = Fovea
        needed_multi_channel_img[:, :, 1] = Disk

        a[0][1].imshow(needed_multi_channel_img)
        a[0][1].set_title('Ground Truth', fontsize = 10)

        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[0][2].imshow(needed_multi_channel_img)
        a[0][2].set_title('Pred. Orig.', fontsize = 10)
        a[0][2].set_xlabel('HBA UNET', fontsize = 10)

        modImgPath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_0_brightness_3.png"
        img, Fovea = process(modImgPath, int(i[1]), int(i[2]), 256)
        a[0][3].imshow(img)
        a[0][3].set_title('Pert. Image', fontsize = 10)
        a[0][3].set_ylabel('Severity: 3', fontsize = 10)
        a[0][3].set_xlabel('Brightness', fontsize = 10)

        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[0][4].imshow(needed_multi_channel_img)
        a[0][4].set_title('Pred. Pert.', fontsize = 10)
        a[0][4].set_xlabel('HBA UNET', fontsize = 10)

"""
#Severity 2

RIGA_BinRushed1_7
#r"C:\\Users\\17033\Desktop\p2OPHAIresults\RIGA_c\FullFundus\\x256-orig\\images\\RIGA_BinRushed1_7.png"

"""
imgpath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_7.png"
for i in df_test:
    if i[0] == "RIGA_BinRushed1_7.png":
        img, Fovea = process(imgpath, int(i[1]), int(i[2]), 256)
        img, Disk = process(imgpath, int(i[3]), int(i[4]), 256)
        a[1][0].imshow(img)
        a[1][0].set_title('Orig. Image', fontsize = 10)

        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = Fovea
        needed_multi_channel_img[:, :, 1] = Disk

        a[1][1].imshow(needed_multi_channel_img)
        a[1][1].set_title('Ground Truth', fontsize = 10)

        print("got here")
        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[1][2].imshow(needed_multi_channel_img)
        a[1][2].set_title('Pred. Orig.', fontsize = 10)
        a[1][2].set_xlabel('UNET', fontsize = 10)

        modImgPath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_7_Temperature_2.png"
        img, Fovea = process(modImgPath, int(i[1]), int(i[2]), 256)
        a[1][3].imshow(img)
        a[1][3].set_title('Pred. Image', fontsize = 10)
        a[1][3].set_ylabel('Severity: 2', fontsize = 10)
        a[1][3].set_xlabel('Temperature', fontsize = 10)

        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[1][4].imshow(needed_multi_channel_img)
        a[1][4].set_title('Pred. Pert.', fontsize = 10)
        a[1][4].set_xlabel('UNET', fontsize = 10)

"""
#Severity 5

RIGA_BinRushed1_2
C:\\Users\\17033\\Desktop\\p2OPHAIresults\\RIGA_c\\FullFundus\\x256-orig\\images\\RIGA_BinRushed1_2.png

"""

imgpath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_2.png"
for i in df_test:
    if i[0] == "RIGA_BinRushed1_2.png":
        img, Fovea = process(imgpath, int(i[1]), int(i[2]), 256)
        img, Disk = process(imgpath, int(i[3]), int(i[4]), 256)
        a[2][0].imshow(img)
        a[2][0].set_title('Orig. Image', fontsize = 10)
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = Fovea
        needed_multi_channel_img[:, :, 1] = Disk

        a[2][1].imshow(needed_multi_channel_img)
        a[2][1].set_title('Ground Truth', fontsize = 10)

        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[2][2].imshow(needed_multi_channel_img)
        a[2][2].set_title('Pred. Orig.', fontsize = 10)
        a[2][2].set_xlabel('UNET', fontsize = 10)

        modImgPath = r"C:\Users\17033\Desktop\example Images\RIGA_BinRushed1_2_impulse_noise_5.png"
        img, Fovea = process(modImgPath, int(i[1]), int(i[2]), 256)
        a[2][3].imshow(img)
        a[2][3].set_title('Pert. Image', fontsize = 10)
        a[2][3].set_ylabel('Severity: 5', fontsize = 10)
        a[2][3].set_xlabel('Impulse Noise', fontsize = 10)
        output = model(np.array([img]))
        output = np.array(output[0])
        needed_multi_channel_img = np.zeros((256, 256, 3))
        needed_multi_channel_img[:, :, 0] = output[:, :, 0]
        needed_multi_channel_img[:, :, 1] = output[:, :, 1]
        a[2][4].imshow(needed_multi_channel_img)
        a[2][4].set_title('Pred. Pert.', fontsize = 10)
        a[2][4].set_xlabel('UNET', fontsize = 10)

fig.savefig("perturbation showcase.png", dpi=600)

