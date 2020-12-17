import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

import scipy.io as sio
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import matplotlib.pyplot as plt
import matplotlib

class YoloModel:
    def __init__(self, weights, config, coco):
        self.weights = weights
        self.config = config
        self.coco = coco
        self.output_layers = []
        self.net = self.load_model()
        self.classes = open(self.coco).read().split("\n")
    
    def load_model(self):
        net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        layer_names = net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net
    
    def transform_bounding_box(self,box, shape=(416,416)):
        
        box = box * np.array([shape[1], shape[0], shape[0], shape[1]]) #width, height, width, height => scale the image
        box = [int(box[0] - (box[2] / 2)), int(box[1] - (box[3] / 2)), int(box[2]), int(box[3])] #xmin, ymin, width, height
        return box

    def get_label_score(self,scores, label="person"):
        label_index = self.classes.index(label)
        return scores[label_index]

    def plot_boxes(self, img, boxes):
        fig,ax = plt.subplots(1) 
        ax.imshow(img)#.permute(1, 2, 0))
        for box in boxes:
            xmin = box[0]
            ymin = box[1]
            w = box[2] 
            h = box[3]
            color = (1,0,1,0.99) 
            rect = matplotlib.patches.Rectangle((xmin,ymin),xmin+w,ymin+h,edgecolor=color,facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def predict(self, img, label="person", threshold=0.70, nms_threshold=0.4,plot=True):
        img = cv2.resize(img, (416,416))
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=True)
        self.net.setInput(blob)


        feed_forward_output = self.net.forward(self.output_layers)
        classes = open(self.coco).read().split("\n")
        person_index = self.classes.index("person")

        boxes = []
        scores = []
        
        for output in feed_forward_output:
            for detection in output: #shape of detection is (85,1) => index 0-4 for bounding box, index 5-85 for scores for each class in coco.names
                label_score = self.get_label_score(detection[5:],label)
                box = detection[0:4] #center X, center Y, width, height
                
                box = self.transform_bounding_box(box, (416,416))
                if(label_score > threshold):
                    boxes.append([box[0],box[1],box[2],box[3]]) #read somewhere that NMSBoxes only takes lists, not np arrays
                    scores.append(float(label_score))

        
        non_overlap_indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, nms_threshold) 
        if(type(non_overlap_indices) is not tuple):
            top_boxes = [boxes[int(i)] for i in non_overlap_indices.flatten()]
            if(plot):
                self.plot_boxes(img,top_boxes)

            return {"boxes": top_boxes, "label_count":len(top_boxes) } 
        else:
            return {"boxes": [], "label_count":0} 

# yolo = YoloModel("yolov3.weights", "yolov3.cfg", "coco.names")
# test = cv2.imread("../crowd_images/mask1.jpg")#, cv2.COLOR_BGR2RGB)
# print(yolo.predict(test)["label_count"])
