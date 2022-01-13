import cv2
import os
import glob
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import csv
from PIL import Image, ImageFont, ImageDraw
import shutil
import numpy as np
import copy
from ensemble_boxes import *

def cv_show(img):
    cv2.imshow('111', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_show(img):
    plt.figure('img')  # 图像窗口名称
    plt.imshow(img)
    plt.show()

def laplacian():
    train_txt = '/Users/videopls/Desktop/AIWIN轮胎检测/train/train_ori.txt'
    # val_txt = '/Users/videopls/Desktop/工业检测/初赛数据/val.txt'

    ori_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train/images/'
    # ori_path = '/Users/videopls/Desktop/工业检测/初赛数据/test/'
    xml_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train/train_annotations/'

    img_save_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train/train_lap/'
    # img_save_path = '/Users/videopls/Desktop/工业检测/初赛数据/lap/test_lap/'

    xml_save_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train/Annotations_lap/'

    save_txt = '/Users/videopls/Desktop/AIWIN轮胎检测/train/train_lap.txt'
    # save_txt = '/Users/videopls/Desktop/工业检测/初赛数据/lap/val_lap.txt'

    lines = open(train_txt, 'r').read().splitlines()
    # lines = open(val_txt, 'r').read().splitlines()
    # with open(save_txt, 'w') as f:
    #     f.write('\n'.join([i+'_lap' for i in lines]))


    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(xml_save_path, exist_ok=True)
    # lines = [i[:-4] for i in os.listdir(ori_path)] # test

    for line in lines:
        img = ori_path + line + '.jpg'

        if line == '01263481_05_BF_00241':
            img = img.replace('.jpg', '.tif')

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # plt_show(img)

        gray_lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        dst = cv2.convertScaleAbs(gray_lap)

        # plt_show(dst)
        cv2.imwrite(img_save_path+line+'_lap.jpg', dst)

        xml = xml_path + line +'.xml'
        shutil.copyfile(xml, xml_save_path+line+'_lap.xml')

if __name__ == '__main__':
    laplacian()