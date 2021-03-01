import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd


from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
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


class FasterRCNN:
    def __init__(self, weight_zip):
        self.weight_zip = weight_zip
        # self.net = 
        # self.classes = 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.num_classes = 4
        self.model = self.load_model()
    
    def load_model(self, eval=True):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) #loads the Faster R-CNN model trained on the COCO dataset
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes) #number of classes = 3 : with_mask, without_mask, mask_not_worn_corrected
        model.load_state_dict(torch.load(self.weight_zip,map_location=self.device))
        if torch.cuda.is_available():
            model.cuda()
        if(eval):
            model.eval()
        return model

    def transform(self, data):
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ]) 
        return tensor_transform(data)

    
    
    def plot_boxes(self, img, annotation, plot_save_to=None):
        fig,ax = plt.subplots(1) 
        # img = img_tensor.cpu().data

        label_mappings = {"with_mask": 3, "mask_weared_incorrect": 2, "without_mask":1} #0 for background class, not used 
        label_tags = ["background","without_mask", "mask_weared_incorrect", "with_mask"]
    
        ax.imshow(img)
        
        for i, box in enumerate(annotation["boxes"]):
            xmin, ymin, xmax, ymax = box
            # print(float(annotation["scores"][i]), label_tags[annotation["labels"][i]])       
            color = (0,1,0,0.1) if (annotation["labels"][i] == 3) else (1,0,0,0.1)
            rect = matplotlib.patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,edgecolor=color,facecolor=color)
            ax.add_patch(rect)
        
        if(plot_save_to):
            plt.savefig(plot_save_to)

        
        plt.show()


    def predict(self, img, threshold=0.75, plot=True, plot_save_to=None):
        img_tensor = self.transform(img)
        img_tensor = img_tensor.to(self.device)
        annotation = self.model([img_tensor])
        boxes = [[box[0].item(),box[1].item(),box[2].item(),box[3].item()] for box in annotation[0]["boxes"]]
        scores = [float(score) for score in annotation[0]["scores"]]
        labels = [int(label) for label in annotation[0]["labels"]]
        non_overlap_indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.4) 
        if(type(non_overlap_indices) is not tuple):
            top_boxes = [boxes[int(i)] for i in non_overlap_indices.flatten()]
            top_scores = [scores[int(i)] for i in non_overlap_indices.flatten()]
            top_labels = [labels[int(i)] for i in non_overlap_indices.flatten()]
            # print("Detected:")
            label_mappings = {3:"with_mask", 2:"mask_weared_incorrect", 1:"without_mask", 0:"background"} #0 for background class, not used 
            string_labels = map(label_mappings.get, top_labels)
            label_frequencies = {"with_mask":0, "mask_weared_incorrect":0,"without_mask":0,"background":0}#Counter(string_labels)
            
            for label in string_labels:
                label_frequencies[label] = label_frequencies.get(label, 0) + 1
            
            # for tup in label_frequencies.most_common():
                # print("\t", tup[0], tup[1])
            top_annotation = {
                "boxes":top_boxes,
                "labels":top_labels,
                "scores":top_scores,
                "label_frequencies":label_frequencies
            }
            if(plot):
                self.plot_boxes(img,top_annotation, plot_save_to)
        else:
            top_annotation = {
                "boxes":[],
                "labels":[],
                "scores":[],
                "label_frequencies":{"with_mask":0, "mask_weared_incorrect":0,"without_mask":0,"background":0}#Counter(string_labels)
            }

        return top_annotation


# rcnn = FasterRCNN("updated_rcnn.zip")
# test = Image.open("../covid_benchmark_data/Train/Mask/0486.jpg")
# rcnn.predict(test,threshold=0.5, plot=True)
