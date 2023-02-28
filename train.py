import argparse

parser = argparse.ArgumentParser(description='Train Landmark model for Project SegLoc (OPHAI).')

parser.add_argument('--model_name', '--model_name', help='Name of model to train.',
                    choices=['unet', 'vggunet', 'resunet', 'hbaunet', 'swinunet', 'hbaunet+attnet', 'unetplusplus',
                             'unetplusplusDE', 'deeplab', 'pix2pix', 'dlib', 'attnet'],
                    required=True)
parser.add_argument('--tr', '--train', help='Name of the CSV file with training dataset information.', required=True)
parser.add_argument('--dd', '--dataset_dir', help='Path to the folder with the CSV files and image subfolders.',
                    required=True)
parser.add_argument('--sp', '--save_path',
                    help='Path to the folder where trained models and all metrics/graphs will be saved.', required=True)

parser.add_argument('--img', '--image_size', type=int,
                    help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--multiprocessing', '--multiprocessing', type=int, help='Amount of Threads to Use')
args = parser.parse_args()

import os
import numpy as np
import tensorflow as tf
import dlib
import multiprocessing
from tqdm import tqdm
from PIL import Image
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import sys
sys.path.insert(0, '/OPHAI-Landmark-Localization-main/ophai')
sys.path.append('../ophai/')
import pandas as pd

print(tf.__version__)

train_path = os.path.join(args.dd, args.tr)
img_size = (args.img, args.img)

dataset_dir = args.dd

df_train = pd.read_csv(train_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
train_paths = []

for r in df_train:
    img_path = os.path.join(os.path.split(dataset_dir)[0], r[1], r[0])
    train_paths.append((img_path, (r[2], r[3]), (r[4], r[5])))

from models.unet import Unet
from models.resunet import ResUnet
from models.vggunet import Vggunet
from models.swinunet import swinunet
from models.hbaunet import hbaunet
from models.uneteffnet import uneteffnet
from models.deeplab import deeplab
from models.attnet import attentionunet
from models.unet2plus import Unet2
import models.pix2pix as Pix2pix
from datautils import process, get_gens

if args.model_name == "swinunet":
    model = swinunet((args.img, args.img, 3), filter_num_begin=16, n_labels=2, depth=6, stack_num_down=4, stack_num_up=4,
                     patch_size=(2, 2), num_heads=[4, 8, 16, 16, 32, 32], window_size=[4, 4, 2, 2, 2, 2], num_mlp=4,
                     output_activation='Sigmoid', shift_window=True,
                     name='swin_unet')
elif args.model_name == "unetplusplus":
    # model = uneteffnet((args.img, args.img, 3), [64, 128, 256, 256, 512], 2, stack_num_down=8, stack_num_up=8,
    #                    activation='ReLU', output_activation='Sigmoid', batch_norm=False, pool=True, unpool=True,
    #                    deep_supervision=False,
    #                    backbone='EfficientNetB4', weights=None, freeze_backbone=True, freeze_batch_norm=True,
    #                    name='xnet')
    model = Unet2(args.img, args.img, color_type=3, num_class=2, deep_supervision=False)
elif args.model_name == "unetplusplusDE":
    model = Unet2(args.img, args.img, color_type=3, num_class=2, deep_supervision=True)
elif args.model_name == "attnet":
    model = attentionunet((args.img, args.img, 3))
elif args.model_name == "hbaunet":
    model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=False, skip=False, num_layers=3)
elif args.model_name == "hbaunet+attnet":
    model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=True, skip=False, num_layers=3)
elif args.model_name == "pix2pix":
    pass
elif args.model_name == "dlib":
    from os.path import exists
    if exists("landmark_localization.xml"):
        output = open("landmark_localization.xml", "w")
        output.write("<?xml version='1.0' encoding='ISO-8859-1'?> \n")
        output.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?> \n")
        output.write("<dataset> \n")
        output.write("<name>Training_Eyes</name> \n")
        output.write("<images> \n")
        for r in tqdm(df_train):
            imagePath = os.path.join(os.path.split(dataset_dir)[0], r[1], r[0])
            print(imagePath)
            if r[1] != "N":
                os.chdir(args.dd)
                img1, Fovea = process(imagePath, int(r[2]), int(r[3]), imgsize = args.img)
                img2, Disk = process(imagePath, int(r[4]), int(r[5]), imgsize = args.img)

                if img1 != "Error" and img2 != "Error":
                    # new_image_path = os.path.basename(imagePath[:imagePath.index(".")]) + ".png"
                    # im = Image.fromarray(img1)
                    # im.save(new_image_path)
                    output.write(f"<image file='{imagePath}'> \n")
                    output.write(f"<box top='1' left='1' width='255' height='255'> \n")
                    coordsf = np.unravel_index(Fovea.argmax(), Fovea.shape)[:2]
                    output.write(f"<part name='Fovea' x='{coordsf[0]}' y='{coordsf[1]}'/> \n")
                    coordsD = np.unravel_index(Disk.argmax(), Disk.shape)[:2]
                    output.write(f"<part name='Disk' x='{coordsD[0]}' y='{coordsD[1]}'/> \n")
                    output.write(f"</box> \n")
                    output.write(f"</image> \n")
        output.write("</images> \n")
        output.write("</dataset> \n")
        #
        options = dlib.shape_predictor_training_options()
        options.tree_depth = 4
        options.nu = 0.1
        options.cascade_depth = 15
        options.feature_pool_size = 400
        options.num_test_splits = 50
        options.oversampling_amount = 5
        options.oversampling_translation_jitter = 0.1
        options.be_verbose = True
        options.num_threads = multiprocessing.cpu_count()
# #
else:
    model = {
        'unet': Unet,
        'resunet': ResUnet,
        'vggunet': Vggunet,
        'swinunet': swinunet,
        'deeplab': deeplab,
    }[args.model_name]((args.img, args.img, 3), 2)


if args.model_name == "dlib":
    print(os.getcwd())
    print("Training Shape Predictor...")
    dlib.train_shape_predictor("landmark_localization.xml", "landmark_predictor_dlib.dat", options)
else:
    os.chdir(args.sp)
    val_size = 0.1
    batch_size = 8
    input_shape = (args.img, args.img, 3)

    train_gen, val_gen, _ = get_gens(args.img, train_paths, [], batch_size, val_size=val_size)
    train_len = int(len(train_paths) * (1 - val_size))
    val_len = len(train_paths) - train_len

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_absolute_error',
                      metrics=['mean_absolute_error', 'accuracy'])
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    history = model.fit(train_gen, validation_data = val_gen, steps_per_epoch=train_len//batch_size, validation_steps = val_len//batch_size, callbacks=[es], verbose = 1, epochs = 200)

    os.chdir(args.sp)

    model.save(args.model_name + ".h5")

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{args.model_name} Loss During Training')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(f'{args.model_name}loss.png', bbox_inches='tight')

    np.save(f'{args.model_name}_TrainingHistory.npy', history.history)