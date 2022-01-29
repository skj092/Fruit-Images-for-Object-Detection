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


class CardsDetector:
    def __init__(self, imagePath, modelPath):
        self.MODEL_NAME = modelPath
        self.IMAGE_NAME = 'uploads/'+imagePath
        print(self.IMAGE_NAME)

    def getPrediction(self):
        img = Image.open(self.IMAGE_NAME)
        img = img.resize((480, 480)).convert('RGB')
        tfms = torchtrans.ToTensor()
        img = tfms(img)
        prediction = self.MODEL_NAME([img])[0]
        nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        print(prediction)
        print('saving image')
        # plot_img_bbox(torch_to_pil(img), nms_prediction)
        print('saving image complete')



        listOfOutput = []
        score = torch.nn.Softmax(prediction['scores'])
        # for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
        #     valDict = {}
        #     valDict["className"] = name
        #     valDict["confidence"] = str(score)
        #     if new_scores is None or new_scores[i] > min_score_thresh:
        #         val = list(new_boxes[i])
        #         valDict["yMin"] = str(val[0])
        #         valDict["xMin"] = str(val[1])
        #         valDict["yMax"] = str(val[2])
        #         valDict["xMax"] = str(val[3])
                    # listOfOutput.append(valDict)
        # plot_img_bbox(torch_to_pil(img), nms_prediction)
        # opencodedbase64 = encodeImageIntoBase64("output.jpg")
        # listOfOutput.append({"image": opencodedbase64.decode('utf-8')})
        return listOfOutput
