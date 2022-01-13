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


def ca_test():
    def contrast_Ratio_brightness(arg):
        # arg参数：为接收新变量地址
        # a为对比度，g为亮度
        # cv2.getTrackbarPos获取滑动条位置处的值
        # 第一个参数为滑动条1的名称，第二个参数为窗口的名称。
        a = cv2.getTrackbarPos(trackbarName1, windowName)
        g = cv2.getTrackbarPos(trackbarName2, windowName)


        # h, w, c = image.shape
        # mask = np.zeros([h, w, c], image.dtype)
        # # cv2.addWeighted函数对两张图片线性加权叠加
        # dstImage = cv2.addWeighted(image, a, mask, 1 - a, g)

        a = a * 0.01
        dstImage = np.uint8(np.clip((a * image + g), 0, 255))

        cv2.imshow("dstImage", dstImage)

    images = glob.glob('/Users/videopls/Desktop/AIWIN轮胎检测/train/images/*.jpg')
    for image in images:
        # image = '/Users/videopls/Desktop/工业检测/初赛数据/train_flip/Ch8DkGBX_luADR_eAAA_pQsmBy4970__3.jpg'
        image = cv2.imread(image)
        cv2.imshow("Saber", image)
        trackbarName1 = "Ratio_a"
        trackbarName2 = "Bright_g"
        windowName = "dstImage"
        a = 200  # 设置a的初值。
        # g = 10  # 设置g的初值。
        g = 0  # 设置g的初值。
        # count1 = 20  # 设置a的最大值
        count1 = 300  # 设置a的最大值
        # count2 = 50  # 设置g的最大值
        count2 = 255  # 设置g的最大值
        # 给滑动窗口命名，该步骤不能缺少！而且必须和需要显示的滑动条窗口名称一致。
        cv2.namedWindow(windowName)

        # 第一个参数为滑动条名称，第二个参数为窗口名称，
        # 第三个参数为滑动条参数，第四个为其最大值，第五个为需要调用的函数名称。
        cv2.createTrackbar(trackbarName1, windowName, a, count1, contrast_Ratio_brightness)
        cv2.createTrackbar(trackbarName2, windowName, g, count2, contrast_Ratio_brightness)
        # 下面这步调用函数，也不能缺少。
        contrast_Ratio_brightness(0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def ca_trans():
    train_txt = '/Users/videopls/Desktop/工业检测/初赛数据/train_all+flip910.txt'
    val_txt = '/Users/videopls/Desktop/工业检测/初赛数据/val.txt'

    # ori_path = '/Users/videopls/Desktop/工业检测/初赛数据/train_all+flip910/'
    ori_path = '/Users/videopls/Desktop/工业检测/初赛数据/test/'
    xml_path = '/Users/videopls/Desktop/工业检测/初赛数据/Annotations_all+flip910/'

    # img_save_path = '/Users/videopls/Desktop/工业检测/初赛数据/ca/train_ca_all/'
    img_save_path = '/Users/videopls/Desktop/工业检测/初赛数据/ca/test_ca/'

    xml_save_path = '/Users/videopls/Desktop/工业检测/初赛数据/ca/Annotations_ca_all/'

    # # lines = open(train_txt, 'r').read().splitlines()
    # lines = open(val_txt, 'r').read().splitlines()
    # # save_txt = '/Users/videopls/Desktop/工业检测/初赛数据/ca/train_ca.txt'
    # save_txt = '/Users/videopls/Desktop/工业检测/初赛数据/ca/val_ca.txt'
    # with open(save_txt, 'w') as f:
    #     f.write('\n'.join([i + '_ca' for i in lines]))

    lines = [i[:-4] for i in os.listdir(ori_path)] # test

    for line in lines:
        img = ori_path + line + '.jpg'

        if line == '01263481_05_BF_00241': # test
            img = img.replace('.jpg', '.tif')

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # plt_show(img)

        a = 1.8  # 设置a的初值。
        g = 0.0  # 设置g的初值。
        dst = np.uint8(np.clip((a * img + g), 0, 255))


        # plt_show(dst)
        cv2.imwrite(img_save_path + line + '_ca.jpg', dst)

        xml = xml_path + line + '.xml'
        shutil.copyfile(xml, xml_save_path + line + '_ca.xml')



if __name__ == '__main__':
    # ca_trans()
    ca_test()