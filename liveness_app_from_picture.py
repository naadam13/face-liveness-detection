import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import time
import cv2
from datetime import datetime

def recognition_liveness(input_path, model_path, le_path, detector_folder, confidence=0.5):
    args = {'input':input_path, 'model':model_path, 'le':le_path, 'detector':detector_folder, 'confidence':confidence}

    print('[INFO] loading face detector')
    proto_path = os.path.sep.join([args['detector'],'deploy.prototxt'])
    model_path = os.path.sep.join([args['detector'],
                                'res10_300x300_ssd_iter_140000.caffemodel'])

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # load the liveness detector model and label encoder from disk
    liveness_model = tf.keras.models.load_model(args['model'])
    le = pickle.loads(open(args['le'], 'rb').read())

    # read the image
    image = cv2.imread(args['input'])
    assert not isinstance(image,type(None)), 'image not found'

    # construct a blob from the image (preprocess image)
    # basically, it does mean subtraction and scaling
    # (104.0, 177.0, 123.0) is the mean of image in FaceNet
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0,
                                (300,300), (104.0, 177.0, 123.0))

    # pass the blob through the NN and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # ensure atleast 1 face it detected
    if len(detections) > 0:
        # we're making the assumption that each image has ONLY ONE face,
        # so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        # ensure that the detection with the highest probability 
        # pass our minumum probability threshold (helping filter out some weak detections)
        if confidence > args['confidence']:
            # compute the (x,y) coordinates of the bounding box
            # for the face and extract face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (32,32))
            face = face.astype('float') / 255.0 
            face = tf.keras.preprocessing.image.img_to_array(face)
            # tf model require batch of data to feed in
            # so if we need only one image at a time, we have to add one more dimension
            # in this case it's the same with [face]
            face = np.expand_dims(face, axis=0)
        
            # pass the face ROI through the trained liveness detection model
            # to determine if the face is 'real' or 'fake'
            # predict return 2 value for each example (because in the model we have 2 output classes)
            # the first value stores the prob of being real, the second value stores the prob of being fake
            # so argmax will pick the one with highest prob
            # we care only first output (since we have only 1 input)
            preds = liveness_model.predict(face)[0]
            j = np.argmax(preds)
            label_name = le.classes_[j] # get label of predicted class
    
    return confidence, label_name

if __name__ == '__main__':
    #name, label_name = recognition_liveness('liveness.model', 'label_encoder.pickle', 
                                            #'face_detector', '../face_recognition/encoded_faces.pickle', confidence=0.5)
    confidence, label_name = recognition_liveness('1.jpg', 'liveness.h5', 'label_encoder.pickle', 
                                            'face_detector', confidence=0.5)
    print(confidence, label_name)