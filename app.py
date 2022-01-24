# importing necessary libraries
from PIL import Image
import torch, torchvision
from glob import glob 
import os
import numpy as np
import matplotlib.pyplot as plt

from com_in_ineuron_ai_utils.utils import decodeImage

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class_names = ['apple', 'banana', 'orange']

# Define a flask app
app = Flask(__name__)



# Function to visualize bounding boxes in the image

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.savefig("output.jpg")


# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')





def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

# Model saved with Keras model.save()
MODEL_PATH = 'model'
# Load your trained model
num_classes = 4 
model = get_object_detection_model(num_classes) 
device = torch.device('cpu')
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path):
    img = Image.open(img_path)
    img = img.resize((480, 480)).convert('RGB')
    tfms = torchtrans.ToTensor()
    img = tfms(img)
    prediction = model([img])[0]
    nms_prediction = apply_nms(prediction, iou_thresh=0.01)
    plot_img_bbox(torch_to_pil(img), nms_prediction)
    listOfOutput = []
    opencodedbase64 = encodeImageIntoBase64("output.jpg")
    listOfOutput.append({"image": opencodedbase64.decode('utf-8')})
    score = torch.nn.Softmax(prediction['scores'])
    return jsonify(listOfOutput)
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        # f = request.files['image']
        image = request.json['image']
        filename = 'inputImage.jpg'
        print(type(image))
        decodeImage(image, filename)


        # Make prediction
        file_path = os.path.join('uploads', "inputImage.jpg")
        print(file_path)
        preds = model_predict(file_path)
        return preds
    return None




if __name__ == '__main__':
    app.run(debug=True)