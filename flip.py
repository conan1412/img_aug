import cv2
import os
import glob
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import csv
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import shutil
import numpy as np
import copy
from ensemble_boxes import *
# import mmcv

def cv_show(img):
    cv2.imshow('111', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_show(img):
    plt.figure('img')  # 图像窗口名称
    plt.imshow(img)
    plt.show()

def change(x, y):
    if x > y:
        tmp = y
        y = x
        x = tmp
    return x, y

# 读取xml，返回[[name, xmin, ymin, xmax, ymax]]的列表
def load_xml(xmlfile):
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    xmlbox = []
    for obj in objs:
        clsname = obj.find('name').text
        xmin = int(obj.find('bndbox')[0].text)
        ymin = int(obj.find('bndbox')[1].text)
        xmax = int(obj.find('bndbox')[2].text)
        ymax = int(obj.find('bndbox')[3].text)
        xmlbox.append([clsname, xmin, ymin, xmax, ymax])
    return xmlbox

def iron_flip1(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path):
    os.makedirs(flip_img_path, exist_ok=True)
    os.makedirs(flip_xml_path, exist_ok=True)

    basenames = open(ori_txt_path, 'r').read().splitlines()
    xmls = glob.glob(ori_xml_path+'/*.xml')
    for xml in tqdm(xmls):
        basename = os.path.basename(xml).strip('.xml')
        if basename not in basenames:
            continue
        ori_img = os.path.join(ori_img_path, basename+'.jpg')

        im = Image.open(ori_img)
        width, height = im.size[0], im.size[1]

        tree = ET.parse(xml)
        root = tree.getroot()

        im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        im1.save(os.path.join(flip_img_path, basename + '__1.jpg'))
        for elem_xmin, elem_ymin, elem_xmax, elem_ymax in zip(root.iter('xmin'), root.iter('ymin'), root.iter('xmax'),
                                                              root.iter('ymax')):
            xmin, ymin, xmax, ymax = int(elem_xmin.text), int(elem_ymin.text), int(elem_xmax.text), int(elem_ymax.text)
            ymin, ymax = change(ymin, ymax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            xmin, xmax = change(xmin, xmax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            xmin , xmax = width-xmax, width-xmin
            elem_xmin.text = str(xmin)
            elem_ymin.text = str(ymin)
            elem_xmax.text = str(xmax)
            elem_ymax.text = str(ymax)
        for elem_filename in root.iter('filename'):
            elem_filename.text = basename + '__1.jpg'
        tree.write(os.path.join(flip_xml_path, basename + '__1.xml'), encoding='utf-8')

def iron_flip2(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path):
    os.makedirs(flip_img_path, exist_ok=True)
    os.makedirs(flip_xml_path, exist_ok=True)

    basenames = open(ori_txt_path, 'r').read().splitlines()
    xmls = glob.glob(ori_xml_path + '/*.xml')
    for xml in tqdm(xmls):
        basename = os.path.basename(xml).strip('.xml')
        if basename not in basenames:
            continue
        ori_img = os.path.join(ori_img_path, basename+'.jpg')

        im = Image.open(ori_img)
        width, height = im.size[0], im.size[1]

        tree = ET.parse(xml)
        root = tree.getroot()

        im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        im2 = im1.transpose(Image.FLIP_TOP_BOTTOM)
        im2.save(os.path.join(flip_img_path, basename + '__2.jpg'))
        for elem_xmin, elem_ymin, elem_xmax, elem_ymax in zip(root.iter('xmin'), root.iter('ymin'), root.iter('xmax'),
                                                              root.iter('ymax')):
            xmin, ymin, xmax, ymax = int(elem_xmin.text), int(elem_ymin.text), int(elem_xmax.text), int(elem_ymax.text)
            ymin, ymax = change(ymin, ymax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            xmin, xmax = change(xmin, xmax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            xmin , xmax = width-xmax, width-xmin
            ymin , ymax = height-ymax, height-ymin
            elem_xmin.text = str(xmin)
            elem_ymin.text = str(ymin)
            elem_xmax.text = str(xmax)
            elem_ymax.text = str(ymax)
        for elem_filename in root.iter('filename'):
            elem_filename.text = basename + '__2.jpg'
        tree.write(os.path.join(flip_xml_path, basename + '__2.xml'), encoding='utf-8')

def iron_flip3(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path):
    os.makedirs(flip_img_path, exist_ok=True)
    os.makedirs(flip_xml_path, exist_ok=True)

    basenames = open(ori_txt_path, 'r').read().splitlines()
    xmls = glob.glob(ori_xml_path + '/*.xml')
    for xml in tqdm(xmls):
        basename = os.path.basename(xml).strip('.xml')
        if basename not in basenames:
            continue
        ori_img = os.path.join(ori_img_path, basename+'.jpg')

        im = Image.open(ori_img)
        width, height = im.size[0], im.size[1]

        tree = ET.parse(xml)
        root = tree.getroot()

        im3 = im.transpose(Image.FLIP_TOP_BOTTOM)
        im3.save(os.path.join(flip_img_path, basename + '__3.jpg'))
        for elem_xmin, elem_ymin, elem_xmax, elem_ymax in zip(root.iter('xmin'), root.iter('ymin'), root.iter('xmax'),
                                                              root.iter('ymax')):
            xmin, ymin, xmax, ymax = int(elem_xmin.text), int(elem_ymin.text), int(elem_xmax.text), int(elem_ymax.text)
            ymin, ymax = change(ymin, ymax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            xmin, xmax = change(xmin, xmax)  # #判断xmin,xmax是否是满足xmax>xmin,不满足则调换顺序
            ymin , ymax = height-ymax, height-ymin
            elem_xmin.text = str(xmin)
            elem_ymin.text = str(ymin)
            elem_xmax.text = str(xmax)
            elem_ymax.text = str(ymax)
        for elem_filename in root.iter('filename'):
            elem_filename.text = basename + '__3.jpg'
        tree.write(os.path.join(flip_xml_path, basename + '__3.xml'), encoding='utf-8')




if __name__ == '__main__':
    ori_img_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train+test500'
    ori_xml_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train+test500'
    ori_txt_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train+test500.txt'

    flip_img_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_flip'
    # flip_img_path = '/Users/videopls/Desktop/工业检测/初赛数据/train_all_flip'
    flip_xml_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/Annotations_flip'
    # flip_xml_path = '/Users/videopls/Desktop/工业检测/初赛数据/Annotations_all_flip'
    flip_txt_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_flip.txt'
    iron_flip1(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path)
    iron_flip2(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path)
    iron_flip3(ori_img_path, ori_xml_path, ori_txt_path, flip_img_path, flip_xml_path, flip_txt_path)
