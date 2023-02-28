import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
os.chdir(r"C:\Users\17033\Desktop\p2OPHAIresults\modelResults")
for i in ['orig']:
    plt.figure(3, figsize=(4, 10))
    hba_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet.h5_{i}_test.csvResults.csv"
    # hba_att_d1 = fr"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet+attnet.h5_{i}_test.csvResults.csv"
    unet_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\unet.h5_{i}_test.csvResults.csv"
    # dlib_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\landmark_predictor_dlib.dat_{i}_test.csvResults.csv"

    hbaunet = pd.read_csv(hba_d1)[['ED_Fovea', 'ED_Disc']].values.tolist()
    # hba_att = pd.read_csv(hba_att_d1)[['ED_Fovea', 'ED_Disc']].values.tolist()
    unet = pd.read_csv(unet_d1)[['ED_Fovea', 'ED_Disc']].values.tolist()
    # dlib = pd.read_csv(dlib_d1)[['ED_Fovea', 'ED_Disc']].values.tolist()

    hbaunet = [(i[0] + i[1])/2 for i in hbaunet]
    # hba_att = [(i[0] + i[1])/2 for i in hba_att]
    unet = [(i[0] + i[1])/2 for i in unet]
    # dlib = [(i[0] + i[1])/2 for i in dlib]

    df = pd.DataFrame(columns = ['Network', 'ED'])
    for i in hbaunet: df = df.append({'Network': 'HBA UNET', 'ED': i},ignore_index = True)
    # for i in hba_att: df = df.append({'Network': 'HBA_ATT_UNET', 'ED': i},ignore_index = True)
    for i in unet: df = df.append({'Network': 'UNET', 'ED': i},ignore_index = True)
    # for i in dlib: df = df.append({'Network': 'DLIB', 'ED': i},ignore_index = True)

    plt.figure(1, figsize=(4, 5), dpi=300)
    ax_orig = sns.boxplot(x="Network", y="ED", data=df,
                 showfliers = False,
                 showmeans=True,
                 meanprops = {"marker": "x", "markerfacecolor": "white",
                              "markeredgecolor": "white"})
    ax_orig.set(ylim=(-.1, 300))

    ax_orig.set_yscale('log', base=10)

    # ax_orig.set_yscale('log')
    ax_orig.set(ylabel='ED (pixels)')
    # ax_orig.suptitle("Without Perturbation", fontweight='bold', fontsize=11)
    ax_orig.text(0.02, 12, "Without Perturbation", color='black', fontsize=12)
    # plt.tight_layout()
    plt.savefig('ED_orig.png', dpi=600, bbox_inches='tight')


#
for i in ['d1']:
    hba_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet.h5_{i}_test.csvResults.csv"
    # hba_att_d1 = fr"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet+attnet.h5_{i}_test.csvResults.csv"
    unet_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\unet.h5_{i}_test.csvResults.csv"
    # dlib_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\landmark_predictor_dlib.dat_{i}_test.csvResults.csv"

    hbaunet = pd.read_csv(hba_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    # hba_att = pd.read_csv(hba_att_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    unet = pd.read_csv(unet_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    # dlib = pd.read_csv(dlib_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()

    preturb001 = [[], [], []]
    preturb010 = [[], [], []]
    preturb011 = [[], [], []]

    df = pd.DataFrame(columns = ['Perturbation', 'Network', 'ED'])

    for i in tqdm(range(len(hbaunet))):
        path = hbaunet[i][0]
        hba_unet_foveaED = hbaunet[i][1]; hba_unet_diskED = hbaunet[i][2]
        # hba_att_unet_foveaED = hba_att[i][1]; hba_att_unet_diskED = hba_att[i][2]
        unet_foveaED = unet[i][1]; unet_diskED = unet[i][2]
        # dlib_foveaED = dlib[i][1]; dlib_diskED = dlib[i][2]

        path = os.path.basename(os.path.normpath(path))
        path = path.split('_')
        preturbation = path[3]
        if preturbation == "001":
            df = df.append({'Perturbation': 'De-illumination', 'Network': 'HBA UNET', 'ED': (hba_unet_foveaED + hba_unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-illumination', 'Network': '  HBA_ATT_UNET', 'ED': (hba_att_unet_foveaED + hba_att_unet_diskED)/2},
        # ignore_index = True)
            df = df.append({'Perturbation': 'De-illumination', 'Network': 'UNET', 'ED': (unet_foveaED + unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-illumination', 'Network': 'DLIB', 'ED': (dlib_foveaED + dlib_diskED)/2},
        # ignore_index = True)
        elif preturbation == "010":
            df = df.append({'Perturbation': 'De-spot', 'Network': 'HBA UNET', 'ED': (hba_unet_foveaED + hba_unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-spot', 'Network': '  HBA_ATT_UNET', 'ED': (hba_att_unet_foveaED + hba_att_unet_diskED)/2},
        # ignore_index = True)
            df = df.append({'Perturbation': 'De-spot', 'Network': 'UNET', 'ED': (unet_foveaED + unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-spot', 'Network': 'DLIB', 'ED': (dlib_foveaED + dlib_diskED)/2},
        # ignore_index = True)
        else:
            df = df.append({'Perturbation': 'De-illumination + De-spot', 'Network': 'HBA UNET', 'ED': (hba_unet_foveaED + hba_unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-illumination + De-spot', 'Network': '  HBA_ATT_UNET', 'ED': (hba_att_unet_foveaED + hba_att_unet_diskED)/2},
        # ignore_index = True)
            df = df.append({'Perturbation': 'De-illumination + De-spot', 'Network': 'UNET', 'ED': (unet_foveaED + unet_diskED)/2},
        ignore_index = True)
        #     df = df.append({'Perturbation': 'De-illumination + De-spot', 'Network': 'DLIB', 'ED': (dlib_foveaED + dlib_diskED)/2},
        # ignore_index = True)

    df.to_csv('d1.csv', index=False)

    df = pd.read_csv('d1.csv')
    print(df.head())
    plt.figure(3, figsize=(5, 6), dpi=300)
    ax3 = sns.catplot(x="Network", y='ED',
                      col="Perturbation", showmeans=True,
                      data=df, kind="box",
                      showfliers=False, height=4, aspect=1,
                      meanprops={"marker": "x", "markerfacecolor": "white",
                                 "markeredgecolor": "white"})
    # ax3.set(ylim=(-.5, 300))
    ax_orig.set(ylim=(-.1, 300))
    ax3.set(yscale="log")
    ax3.set(ylabel='ED (pixels)')
    # ax3.fig.suptitle("Perturbation Types", fontweight='bold', fontsize=11)
    ax3.fig.text(0.45, 1, "Perturbation Types", color='black', fontsize=12)
    # plt.tight_layout()
    plt.savefig('ED_d1.png', dpi=600, bbox_inches='tight')
    plt.show()

for i in ['d2']:
    hba_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet.h5_{i}_test.csvResults.csv"
    # hba_att_d1 = fr"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\hbaunet+attnet.h5_{i}_test.csvResults.csv"
    unet_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\unet.h5_{i}_test.csvResults.csv"
    # dlib_d1 = rf"C:\Users\17033\Desktop\p2OPHAIresults\modelResults\landmark_predictor_dlib.dat_{i}_test.csvResults.csv"

    hbaunet = pd.read_csv(hba_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    # hba_att = pd.read_csv(hba_att_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    unet = pd.read_csv(unet_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    # dlib = pd.read_csv(dlib_d1)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()

    df = pd.DataFrame(columns = ['Perturbation', 'Network', 'ED', 'Severity'])

    for i in tqdm(range(len(hbaunet))):
        path = hbaunet[i][0]
        hba_unet_foveaED = hbaunet[i][1]; hba_unet_diskED = hbaunet[i][2]
        # hba_att_unet_foveaED = hba_att[i][1]; hba_att_unet_diskED = hba_att[i][2]
        unet_foveaED = unet[i][1]; unet_diskED = unet[i][2]
        # dlib_foveaED = dlib[i][1]; dlib_diskED = dlib[i][2]
        path = os.path.basename(os.path.normpath(path))
        for k in ['gaussian_noise', 'Hue_minus', 'Hue_plus', 'impulse_noise', 'jpeg_compression', 'Saturation_minus',
                  'Saturation_plus', 'shot_noise', 'speckle_noise']:
            if k in path:
                path = path.replace(k, k.replace("_", " ")).title()
                break

        path = path.split('_')
        preturbation = path[3]
        severity = path[4]
        severity = severity[0]
        if preturbation != "Temperature":
            df = df.append({'Perturbation': preturbation.replace('_', ' ').title(),'Severity': severity, 'Network': 'HBA_UNET'.replace('_', ' ').title(), 'ED': (hba_unet_foveaED + hba_unet_diskED)/2},
            ignore_index = True)
            # df = df.append({'Perturbation': preturbation,'Severity': severity, 'Network': 'HBA_ATT_UNET', 'ED': (hba_att_unet_foveaED + hba_att_unet_diskED)/2},
            # ignore_index = True)
            df = df.append({'Perturbation': preturbation.replace('_', ' ').title(),'Severity': severity, 'Network': 'UNET'.replace('_', ' ').title(), 'ED': (unet_foveaED + unet_diskED)/2},
            ignore_index = True)
            # df = df.append({'Perturbation': preturbation,'Severity': severity, 'Network': 'DLIB', 'ED': (dlib_foveaED + dlib_diskED)/2},
            # ignore_index = True)
    print(df.head())
    print(df.Perturbation.unique())
    print(df.Severity.unique())
    print(df.Network.unique())

    df.to_csv('d2.csv', index=False)
    df = pd.read_csv('d2.csv')


    hue_order = [1, 2, 3, 4, 5]
    col_category = 'Perturbation Type'

    plt.figure(3, figsize=(4, 5), dpi=600)
    ax3 = sns.catplot(x='Network', y='ED',
                      hue='Severity', col="Perturbation", showmeans=True,
                      data=df, kind="box", hue_order=hue_order,
                      showfliers=False, height=4, aspect=1, col_wrap =4,
                      meanprops={"marker": "x", "markerfacecolor": "white",
                                 "markeredgecolor": "white"})
    # ax.set_yscale('log')
    ax3.set(yscale="log")

    ax3.set_titles("{col_name}")
    ax3.set(ylabel='ED (pixels)')
    # ax3.fig.suptitle("Perturbation Types", fontweight='bold', fontsize=11)
    ax3.fig.text(0.45, 1, "Perturbation Types", color='black', fontsize=12)
    # plt.tight_layout()
    plt.savefig('ED_d2.png', dpi=300, bbox_inches='tight')
#

#
