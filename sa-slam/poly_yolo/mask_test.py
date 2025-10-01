

from datetime import datetime
import colorsys
import os
import sys
from functools import reduce
from functools import wraps

import math
import random as rd
import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

MAX_VERTICES = 1000 
img_out_path   = '~/wh4/img/'
out_path       = '~/wh4/test_anno/' #path, where the images will be saved. The path must exist

annotation_path = '~/wh4/wh-train.txt'
validation_path = '~/wh4/wh-val.txt'


#helper function
def translate_color(cls):
    if cls == 0: return (0, 128, 128)
    if cls == 1: return (128, 0, 128)
    if cls == 2: return (255, 225, 25)
    if cls == 3: return (0, 130, 200)
    if cls == 4: return (245, 130, 48)
    if cls == 5: return (145, 30, 180)
    if cls == 7: return (70, 240, 240)
    if cls == 8: return (240, 50, 230)
    if cls == 9: return (210, 245, 60)
    if cls == 10: return (250, 190, 190)
    if cls == 11: return (230, 25, 75)
    if cls == 12: return (230, 190, 255)
    if cls == 13: return (170, 110, 40)
    if cls == 14: return (255, 250, 200)
    if cls == 15: return (60, 180, 75)
    if cls == 16: return (170, 255, 195)
    if cls == 17: return (128, 128, 0)
    if cls == 18: return (255, 215, 180)
    if cls == 19: return (80, 80, 128)

def boxed_from_lines(lines):
    imgs = 0
    total_boxes = 0
    for im in range (0, len(lines)):
        imgs    += 1
        line = lines[im].split()
        img  = cv.imread(line[0])
        ori_img = img.copy()
        overlay = img.copy()
        num_boxes = len(line)-1
        box = []
        boxes   = []
        classes = []
        polygons = []
        for element in range(1, len(line)):
            box_n_polygon = line[element].split(',')
            box.append(box_n_polygon[0:4])
            classes.append(box_n_polygon[4])
            polygons.append(box_n_polygon[5:len(box_n_polygon)])
        
        #example, hw to reshape reshape y1,x1,y2,x2 into x1,y1,x2,y2
        for k in range (0, len(box)):
            top_left = ()
            boxes.append((box[k][1], box[k][0], box[k][3], box[k][2]))
            cv.rectangle(img, (int(box[k][0]), int(box[k][3])), (int(box[k][2]),int(box[k][1])), translate_color(int(classes[k])), 3, 1)
        total_boxes += len(boxes)
        
        #browse all boxes
        for b in range(0, len(boxes)):
            f              = translate_color(int(classes[b]))    
            points_to_draw = []

            for dst in range(0, len(polygons[b])//2):
                points_to_draw.append([int(polygons[b][dst*2]), int(polygons[b][dst*2+1])])
            
            
            points_to_draw = np.asarray(points_to_draw)
            points_to_draw = points_to_draw.astype(np.int32)
            if points_to_draw.shape[0]>0:
                cv.polylines(img, [points_to_draw],True,f, thickness=2)
                cv.fillPoly(overlay, [points_to_draw], f)
        img = cv.addWeighted(overlay, 0.4, img, 1 - 0.4, 0)
        imgpath = line[0].split('/')

        cv.imwrite(img_out_path+str(imgpath[5]), ori_img)
        cv.imwrite(out_path+str(imgpath[5]), img)
        
    print('total boxes: ', total_boxes)
    print('imgs: ', imgs)

def _main():
    

    with open(annotation_path) as f:
        lines = f.readlines()
    boxed_from_lines(lines)


if __name__ == '__main__':
    _main()