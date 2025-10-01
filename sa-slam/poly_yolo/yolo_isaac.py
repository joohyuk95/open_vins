# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


"""Helper class for writing groundtruth data offline in kitti format.
"""

from cmath import pi, sin
import csv
from ctypes import sizeof
import os

import matplotlib.pyplot as plt
from PIL import Image
from construct import Numpy
from .base import BaseWriter
import carb
from numpy import linspace 
from numpy import zeros
from numpy import size as npsize
from numpy import asarray as npasarray
from numpy import concatenate as npconcatenate
from numpy import ones as npones
from numpy import zeros as npzeros
from numpy import hstack as nphstack
from numpy import vstack as npvstack
from random import randint
import cv2


class YoloWriter(BaseWriter):
    


    def __init__(
        self,
        data_dir="kitti_data",
        num_worker_threads=4,
        max_queue_size=500,
        train_size=10,
        classes=[],
        bbox_type="BBOX2DLOOSE",
    ):
        BaseWriter.__init__(self, data_dir, num_worker_threads, max_queue_size)
        self.create_output_folders()
        self.train_size = train_size
        self.classes = classes
        self.bbox_type = bbox_type
        self.class_mapping = {'forklifts': '0',
                     'worker': '1',
                     }
        if self.bbox_type is not "BBOX2DLOOSE" and self.bbox_type is not "BBOX2DTIGHT":
            carb.log_error(
                f"bbox_type must be BBOX2DLOOSE or BBOX2DTIGHT, it is currently set to {self.bbox_type} which is not supported, defaulting to BBOX2DLOOSE"
            )
            self.bbox_type = "BBOX2DLOOSE"

    def worker(self):
        """Processes task from queue. Each tasks contains groundtruth data and metadata which is used to transform the output and write it to disk."""
        while True:
            data = self.q.get()
            if data is None:
                break
            else:
                self.save_image(data)
                if int(data["METADATA"]["image_id"]) < self.train_size:
                    label, isPass = self.get_label_line(data)
                    if(isPass): self.resQ.put(label)
                else:
                    label, isPass = self.get_label_line(data)
                    if(isPass): self.resQ.put(label)
            self.q.task_done()

    def writer(self):
        cnt = 0
        with open(os.path.join(self.data_dir, "wh-train.txt"), "w") as train_file, open(os.path.join(self.data_dir, "wh-val.txt"), "w") as val_file:
            while True:
                result = self.resQ.get()
                if (cnt < self.train_size): print(result, file=train_file)
                else: print(result, file=val_file)
                self.resQ.task_done()
                #print(cnt)
                cnt += 1

    def get_label_line(self, data):
        """Saves the labels for the 2d bounding boxes in Kitti format."""
        label_set = []
        viewport_width = data["METADATA"][self.bbox_type]["WIDTH"]
        viewport_height = data["METADATA"][self.bbox_type]["HEIGHT"]

        outputlineFile = os.path.join(self.train_folder, f"{data['METADATA']['image_id']}.png") + ' '
        outputlineData = ''
        isPass = False
        self.wpadding = 4
        # determine polyline 
        seg_img = []
        for gt_type, sdata in data["DATA"].items():
            if gt_type == "SEMANTIC":
                seg_img = sdata

        for box in data["DATA"][self.bbox_type]:
            semantic_label = str(box[2])
            semantic_id = box[5]
            #Skip label if not in class list
            if self.classes != [] and semantic_label not in self.classes:
                continue
            # 2D bounding box points
            x_min, y_min, x_max, y_max = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            
            # Check if bounding boxes are in the viewport
            if (
                x_min < 0
                or y_min < 0
                or x_max > viewport_width
                or y_max > viewport_height
                or x_min > viewport_width
                or y_min > viewport_height
                or y_max < 0
                or x_max < 0
            ):
                continue
            
            isPass = True
            
            outputlineData = outputlineData + str(round(x_min)) + ',' + \
                                            str(round(y_min)) + ',' + \
                                            str(round(x_max)) + ',' + \
                                            str(round(y_max)) + ','

            if semantic_label in self.class_mapping:
                outputlineData = outputlineData + self.class_mapping[semantic_label] + ','
                
            else:
                print('Error! Undefined class ', semantic_label, ' in picture ', f"{data['METADATA']['image_id']}.png") 

            # **************** Setting up polyline*****************
            box_center = [int((x_max-x_min)/2),int((y_max-y_min)/2)]
            box_seg = seg_img[y_min:y_max, x_min:x_max]
            
            if(npsize(box_seg)):
                ang = linspace(0,360,36)
                rs = zeros(36)
                box_seg[box_seg != semantic_id] = 0
                box_seg[box_seg == semantic_id] = 255
                # box_seg[box_seg != semantic_id] = 0
                #print(box_seg)
                
                cnt, contour = self.get_single_centerpoint(box_seg, semantic_id,data['METADATA']['image_id'])
                if(cnt[0] == -1):
                    # No polyline available in the current data set. Just copy box2d information.
                    outputlineData = outputlineData + str(round(x_min)) + ',' + \
                                                    str(round(y_min)) + ',' + \
                                                    str(round(x_max)) + ',' + \
                                                    str(round(y_max)) + ' '
                    return  " ", False
                else:
                    polygon = contour
                    print(polygon.shape)
                    polygon[:,:,0] += (x_min - self.wpadding)  
                    polygon[:,:,1] += (y_min - self.wpadding) 
                    polygon = polygon.astype(int)
                    for i in range(len(polygon)):
                        if(i == len(polygon)-1):
                            outputlineData = outputlineData + str(round(polygon[i][0][0])) + ',' + str(round(polygon[i][0][1])) + ' ' 
                        else:
                            outputlineData = outputlineData + str(round(polygon[i][0][0])) + ',' + str(round(polygon[i][0][1])) + ',' 
                    #print(semantic_label, semantic_id)
                    #print(box_seg[box_center[1],box_center[0]])
            
        label = outputlineFile + outputlineData[:-1]
        #outputlineData = ''
        return  label, isPass

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def get_single_centerpoint(self, mask, semantic_id, image_name):
        # add a padding to the mask
        if(self.wpadding):
            side_pad = npzeros((mask.shape[0], self.wpadding))
            mask = nphstack((side_pad,mask,side_pad))
            top_pad = npzeros((self.wpadding,mask.shape[1]))
            mask = npvstack((top_pad,mask,top_pad))

        mask = npasarray(mask, dtype="uint8")
        if(True):
            cv2.imwrite('~/output_yolo_seg/masks/'+str(image_name)+'_'+str(randint(0, 10))+'.jpg', mask)

        all_contours = False
        image, contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contour)>0):
            if all_contours:
                contour = [npconcatenate(contour, axis=0)] # combine all contours yielding largest mask
            else:     
                contour.sort(key=lambda x: cv2.contourArea(x), reverse=True)  # only save the biggest contour

            t_contour = contour[0]
            count = t_contour[:, 0, :]
            try:
                center = self.get_centerpoint(count)
            except:
                x,y = count.mean(axis=0)
                center=[int(x), int(y)]
            
            if(len(t_contour)<10):
                return [-1, -1], False
                
            #decrease the number of contour, to speed up
            # 360 points should ok, the performance drop very tiny.
            max_points = 360
            if len(t_contour) > max_points:
                compress_rate = len(t_contour) // max_points
                t_contour = t_contour[::compress_rate]
            return center, t_contour
        else:
            return [-1, -1], False

    def save_image(self, data):
        rgb_img = Image.fromarray(data["DATA"]["RGB"], "RGBA").convert("RGB")
        rgb_img.save(f"{self.train_folder}/{data['METADATA']['image_id']}{'.png'}")

    def create_output_folders(self):
        """Checks if the output folders are created. If not, it creates them."""
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.train_folder = os.path.join(self.data_dir, "imgs")
        #self.test_folder = os.path.join(self.data_dir, "testing")

        if not os.path.exists(self.train_folder):
            os.mkdir(self.train_folder)
