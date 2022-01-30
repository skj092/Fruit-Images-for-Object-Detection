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
class_names = ['_','apple', 'banana', 'orange']
from flask_cors import CORS, cross_origin
from obj import CardsDetector

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

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
model.eval()
print('Model loaded. Check http://127.0.0.1:5000/')



# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        modelPath = model
        self.objectDetection = CardsDetector(self.filename, modelPath)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.objectDetection.getPrediction()
    return jsonify(result)

# @app.after_request
# def add_headers(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     return response


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=7000, debug=True)
