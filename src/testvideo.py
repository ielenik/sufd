import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import json
from math import cos, sin
from random import shuffle
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
model = tf.keras.models.load_model('best.h5')
pad_size = 64
threshold = 0.1

def find_faces(img):
    if len(img.shape) == 2:
        img = np.stack( [img,img,img], axis = -1)

    w = img.shape[1]
    h = img.shape[0]
    scale = 1

    osth = (pad_size*1000-h)%pad_size
    ostw = (pad_size*1000-w)%pad_size
    if osth != 0 or ostw != 0:
        img = np.pad(img, ((0,osth),(0,ostw),(0,0)))
        h += osth
        w += ostw
    
#    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    img = img.astype(np.float32)/127.5-1


    img = np.stack( [img,img,img,img,img,img,img,img,img,img,img,img,img,img,img,img,], axis = 0)
    pr = model(img)[0:1]

    centers = pr[:,:,:,0:1]
    centers_mp = tf.nn.max_pool2d(centers, ksize=(3, 3), strides=(1, 1), padding="SAME")[:,:,:,0]
    centers = centers[:,:,:,0]
    faces = tf.logical_and(tf.greater(tf.nn.sigmoid(centers), threshold), tf.equal(centers, centers_mp))
    centers = tf.where(faces)
    faces = tf.gather_nd(pr, centers)
    num_faces = len(faces)

    img_cv = (((img+1)*127.5)[0]).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    faces = faces.numpy()
    centers = centers.numpy()

    for j in range(num_faces):
    # centers = tf.where(faces)
    # faces = tf.greater(pr, centers)
    # num_faces = len(faces)
        a = np.exp(faces[j,3] + 3.5)
        angle = faces[j,4]
        centre_x = int((centers[j,2] + faces[j,1])*16)
        centre_y = int((centers[j,1] + faces[j,2])*16)

        p1 = str(int(sigmoid(faces[j,5])*1000)/10)
        #p2 = str(int(sigmoid(faces[j,4])*1000)/10)
        cv2.putText(img_cv, p1, (centre_x - 25,centre_y - 25), font, 1, (0, 255, 0))            
        #cv2.putText(img_cv, p2, (centre_x - 25,centre_y), font, 1, (0, 255, 0))            
        for j in range(4):
            p1s = (a,-a)
            p2s = (a,+a)

            p1d = ( int(cos(angle)*p1s[0]+sin(angle)*p1s[1]+centre_x), int(-sin(angle)*p1s[0]+cos(angle)*p1s[1]+centre_y))
            p2d = ( int(cos(angle)*p2s[0]+sin(angle)*p2s[1]+centre_x), int(-sin(angle)*p2s[0]+cos(angle)*p2s[1]+centre_y))

            cv2.line(img_cv, p1d, p2d, (0,255,0), 1)
            angle += math.pi/2

    # _, ax = plt.subplots(1, 1)
    # ax.imshow(np.array(img_cv))
    # plt.show()
    return img_cv

import time

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('77677667.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

total_frames = -50
total_time = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, dsize=(256, int(frame.shape[0]*256/frame.shape[1])), interpolation=cv2.INTER_LINEAR)
    
    if total_frames == -1:
        total_frames = 0
        total_time = 0

    total_time -= time.perf_counter()
    frame = find_faces(frame)
    total_time += time.perf_counter()
    total_frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

    print(total_time/total_frames, end = '\r')
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break    