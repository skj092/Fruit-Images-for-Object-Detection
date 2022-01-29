# Author: Sourangshu Pal
# Date: 15/11/19
# Import packages
import os
import numpy as np
from com_in_ineuron_ai_utils.utils import encodeImageIntoBase64
from utils import plot_img_bbox
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision import transforms as torchtrans
from utils import apply_nms, plot_img_bbox, torch_to_pil
import torch
import cv2


class CardsDetector:
    def __init__(self, imagePath, modelPath):
        self.MODEL_NAME = modelPath
        self.IMAGE_NAME = 'uploads/'+imagePath
        print(self.IMAGE_NAME)

    def getPrediction(self):
        image = cv2.imread(self.IMAGE_NAME)
        image = cv2.resize(image, (480, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tfms = torchtrans.ToTensor()
        img = tfms(image)
        prediction = self.MODEL_NAME([img])[0]
        nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        print(nms_prediction)
        print('saving image')
        # plot_img_bbox(torch_to_pil(img), nms_prediction)
        

        listOfOutput = []
        score = torch.nn.Softmax(prediction['scores'])
        valDict = {}
        valDict["confidence"] = str(score)
        listOfOutput.append(valDict)
        # drawing box on image

        for box in nms_prediction['boxes']:
            box = box.detach().numpy()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            img = cv2.rectangle(image, (x1, y1), (x2, y2), (225,0,0), 3)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('output.jpg', image)
        print('saving image complete')

        opencodedbase64 = encodeImageIntoBase64("output.jpg")
        listOfOutput.append({"image": opencodedbase64.decode('utf-8')})
        return listOfOutput
