import argparse

parser = argparse.ArgumentParser(description='Testing Landmark Localization model for Project SegLoc (OPHAI).')

parser.add_argument('--tr', '--test', help='Name of the CSV file with testing dataset information.', required=True)
parser.add_argument('--dd', '--dataset_dir', help='Path to the folder with the CSV files and image subfolders.',
                    required=True)
parser.add_argument('--path_trained_model', '--path_trained_model',
                    help='Path to the numpy array with the train files and image subfolders.',
                    required=False)
parser.add_argument('--sp', '--save_path',
                    help='Path to the folder where trained models and all metrics/graphs will be saved.',
                    required=True)
parser.add_argument('--img', '--image_size', type=int,
                    help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)

args = parser.parse_args()

import os
import numpy as np
import tensorflow as tf
import dlib
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datautils import process
import csv
from sklearn.metrics import mean_squared_error
from models.hbaunet import hbaunet
from models.hbaunet import HBA

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
np_config.enable_numpy_behavior()

print(tf.__version__)



def ED(x, gt):
    prediction = np.unravel_index(x.argmax(), x.shape)
    groundtruth = np.unravel_index(gt.argmax(), gt.shape)
    dist = np.linalg.norm(np.asarray([prediction[0], prediction[1]]) - np.asarray(groundtruth))
    return dist, prediction[0], prediction[1], groundtruth[0], groundtruth[1]


def rmse(predictions, targets):
    return mean_squared_error(predictions, targets, squared=False)

def EDdat(loc, gt):
    groundtruth = np.unravel_index(gt.argmax(), gt.shape)
    dist = np.linalg.norm(np.asarray([loc[0], loc[1]]) - np.asarray(groundtruth))
    return dist, loc[0], loc[1], groundtruth[0], groundtruth[1]

# GET PREDICTION BASED RESUlTS FOR BOXPLOTS

for i in ["_d2_test.csv", "_d1_test.csv",  "_orig_test.csv",  "_d4_test.csv", "_d5_test.csv"]:
    if ".dat" in args.path_trained_model:
        model = dlib.shape_predictor(args.path_trained_model)
    elif "hbaunet" in args.path_trained_model and "attnet" in args.path_trained_model:
        model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=True, skip=False, num_layers=3)
        model.load_weights(args.path_trained_model)
    elif "hbaunet" in args.path_trained_model:
        model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=False, skip=False, num_layers=3)
        model.load_weights(args.path_trained_model)

    else:
        model = tf.keras.models.load_model(args.path_trained_model)

    os.chdir(args.dd)

    train_path = args.tr + i
    df_test = pd.read_csv(train_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
    foveaEDbin = []
    diskEDbin = []
    FoveaPreds = []
    DiskPreds = []
    FoveaGts = []
    DiskGts = []

    out_write = open(args.path_trained_model + i + 'Results.csv', 'w')
    csvwriter = csv.writer(out_write)

    fields = ['img_path', 'gt_Fovea_X', 'gt_Fovea_Y', 'gt_Disc_X', 'gt_Disc_Y', 'p_Fovea_X', 'p_Fovea_Y', 'p_Disc_X',
              'p_Disc_Y', "ED_Fovea", "ED_Disc"]
    rows = []
    csvwriter.writerow(fields)

    os.chdir(args.sp)

    for r in tqdm(df_test):
        try:
            img_path = os.path.join(os.path.split(args.dd)[0], r[1], r[0])
            img, Fovea = process(img_path, int(r[2]), int(r[3]), args.img)
            img, Disk = process(img_path, int(r[4]), int(r[5]), args.img)
            if type(img) != list:
                if ".dat" in args.path_trained_model:
                    dets = dlib.rectangle(left=1, top=1, right=255, bottom=255)
                    output = model(np.array(img), dets)
                    foveaLoc = (output.part(0).x, output.part(0).y)
                    diskLoc = (output.part(1).x, output.part(1).y)
                    EDFovea, fpX, fpY, fgX, fgY = EDdat(foveaLoc, Fovea)
                    EDDisk, dpX, dpY, dgX, dgY = EDdat(diskLoc, Fovea)
                else:
                    output = model(np.array([img]))
                    output = np.array(output[0])
                    EDFovea, fpX, fpY, fgX, fgY = ED(output[:, :, 0], Fovea)
                    EDDisk, dpX, dpY, dgX, dgY = ED(output[:, :, 1], Disk)
                row = [img_path, fgX, fgY, dgX, dgY, fpX, fpY, dpX, dpY, EDFovea, EDDisk]
                rows.append(row)
                foveaEDbin.append(EDFovea)
                diskEDbin.append(EDDisk)
                FoveaPreds.append((fpX, fpY))
                DiskPreds.append((dpX, dpY))
                FoveaGts.append((fgX, fgY))
                DiskGts.append((dgX, dgY))
        except FileNotFoundError:
            continue
        except RuntimeError:
            continue
        except OSError:
            continue
        except ValueError:
            continue
    csvwriter.writerows(rows)

    # GET COMPACT RESULTS
    compact_out_write = open(args.path_trained_model + i + 'CompactResults.csv', 'w', newline='')
    csvwriter = csv.writer(compact_out_write)
    fields = ['Avg_ED_Fovea', 'Avg_ED_Disk', 'RMSE_Fovea', 'RMSE_Disk']
    Avg_ED_Fovea = sum(foveaEDbin) / len(foveaEDbin)
    Avg_ED_Disk = sum(diskEDbin) / len(diskEDbin)
    RMSE_Fovea = rmse(FoveaPreds, FoveaGts)
    RMSE_Disk = rmse(DiskPreds, DiskGts)
    row = [str(Avg_ED_Fovea), str(Avg_ED_Disk), str(RMSE_Fovea), str(RMSE_Disk)]
    csvwriter.writerow(fields)
    csvwriter.writerow(row)

