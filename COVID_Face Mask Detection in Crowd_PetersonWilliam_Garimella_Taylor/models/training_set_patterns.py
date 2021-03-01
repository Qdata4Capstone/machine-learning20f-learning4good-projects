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


if __name__ == "__main__":

    stack = ModelStack()
    base_dir = "../training_rcnn_images/images"
    cols = ["Image_Path","RCNN_Person", "Mask", "Incorrect_Mask", "Without_Mask", "YOLO_Person", "Minimum_Compliance", "Maximum_Compliance"]
    predictions = {'Image_Path': [], 'RCNN_Person': [], 'Mask': [], 'Incorrect_Mask': [], 'Without_Mask': [], 'YOLO_Person': [], 'Minimum_Compliance': [], 'Maximum_Compliance': []}
    len_dir = len(os.listdir(base_dir)) -1
    index = 1
    for path in os.listdir(base_dir):
        if(path != ".DS_Store"):
            print(index, "\\", len_dir)
            index += 1
            img = cv2.cvtColor(cv2.imread(os.path.join(base_dir,path)), cv2.COLOR_BGR2RGB)
            compliance = stack.get_mask_compliance(img,img_path=path)
            for col in cols:
                predictions[col].append(compliance[col])
            
        

            
    # print(predictions)
    df = pd.DataFrame(data=predictions)
    print(df)
    df.to_csv("../training_rcnn_images/training_predictions.csv",index=False)
    


