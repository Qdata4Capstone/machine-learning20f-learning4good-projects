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
from torchvision import transforms, datasets, models
from rcnn_model import FasterRCNN
import time


rcnn = FasterRCNN("updated_rcnn.zip")
test = cv2.imread("test.jpg")#, cv2.COLOR_BGR2RGB)


frame_count = 0
frames = []
capture = cv2.VideoCapture(0)  
step = 40  
while True:
    ret, frame = capture.read()
    
    if not ret:
        break        
    
    cv2.imshow('frame',frame)
    timer = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        start = time.time()
        
        # cv2.imshow('frame',frame)
        # cv2.putText(frame,"3", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        while(timer < 3):
            end = time.time()
            timer = end - start

        ret, frame = capture.read()
        cv2.imshow('frame',frame)

        # cv2.putText(frame,"2", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        # time_count + = time.time() - time_count
        # cv2.putText(frame,"1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        # time_count + = time.time() - time_count
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        break

    # frame_count += 1
    # if(frame_count % step == 0):
        # rcnn.predict(frame,threshold=0.75)







# for frame in frames:
#   frame_w_boxes = yolo_segment(frame, net)
#   plt.imshow(frame_w_boxes)
#   plt.show()
#   # print(type(frame_w_boxes))
#   # video.write(frame_w_boxes)  

capture.release()
cv2.destroyAllWindows()

print(rcnn.predict(converted,threshold=0.75,plot=False))

