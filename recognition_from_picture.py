import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import face_recognition
import tensorflow as tf
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import time
import cv2
from datetime import datetime

def recognition(input_path, detector_folder, encodings, confidence):
    args = {'input':input_path, 'detector':detector_folder, 'encodings':encodings, 'confidence':confidence}

    # load the encoded faces and names
    print('[INFO] loading encodings...')
    with open(args['encodings'], 'rb') as file:
        encoded_data = pickle.loads(file.read())
    # load our serialized face detector from disk
    print('[INFO] loading face detector')
    proto_path = os.path.sep.join([args['detector'],'deploy.prototxt'])
    model_path = os.path.sep.join([args['detector'],
                                'res10_300x300_ssd_iter_140000.caffemodel'])

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # read the image
    image = cv2.imread(args['input'])

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
            # expand the bounding box so that the model can recog easier
            face_to_recog = face # for recognition
            face = cv2.resize(face, (32,32))
            # face recognition
            rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
            #rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            # initialize the default name if it doesn't found a face for detected faces
            name = 'Unknown'
            # loop over the encoded faces (even it's only 1 face in one bounding box)
            # this is just a convention for other works with this kind of model
            for encoding in encodings:
                matches = face_recognition.compare_faces(encoded_data['encodings'], encoding) 
                # check whether we found a matched face
                if True in matches:
                    # find the indexes of all matched faces then initialize a dict
                    # to count the total number of times each face was matched
                    matchedIdxs = [i for i, b in enumerate(matches) if b]
                    counts = {}
                    # loop over matched indexes and count
                    for i in matchedIdxs:
                        name = encoded_data['names'][i]
                        counts[name] = counts.get(name, 0) + 1      
                    # get the name with the most count
                    name = max(counts, key=counts.get)
    
    return confidence, name

if __name__ == '__main__':
    #name, label_name = recognition_liveness('liveness.model', 'label_encoder.pickle', 
                                            #'face_detector', '../face_recognition/encoded_faces.pickle', confidence=0.5)
    confidence, name = recognition('dataset_image/bill_gates.jpg', 'face_detector', '../face_recognition/encoded_faces.pickle', confidence=0.5)
    print(confidence, name)