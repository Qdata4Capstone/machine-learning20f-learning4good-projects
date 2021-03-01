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
from yolo_model import YoloModel
from rcnn_model import FasterRCNN
from crowd_analysis import ModelStack
from collections import Counter

if __name__ == "__main__":

    stack = ModelStack()
    base_dir = "../covid_benchmark_data/train/NonMask"
    len_dir = len(os.listdir(base_dir)) -1
    index = 0
    person_count = 0
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            index += 1
            print(index, "/", len_dir)
            person_count += stack.get_person_count(img,threshold=0.9,nms_threshold=0.1,plot=False)
            
    
    print("Nonmask Person Count: ",person_count)
    
    index = 0
    person_count = 0
    base_dir = "../covid_benchmark_data/train/Mask"
    len_dir = len(os.listdir(base_dir)) -1
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            index += 1
            print(index, "/", len_dir)
            person_count += stack.get_person_count(img,threshold=0.9,nms_threshold=0.1,plot=False)
            
    print("Mask Person Count: ",person_count)
            
            
#YOLO:
# Nonmask Person Count:  99
# Mask Person Count:  103
