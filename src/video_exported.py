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
model1 = tf.keras.models.load_model('models/best.h5.h5')
model2 = tf.keras.models.load_model('models/find_face.h5')
model3 = tf.keras.models.load_model('models/best.h5')
pad_size = 64
threshold = 0.5
MULTIPLE = 64
FRAMES_AVG = 16

models = [ model1,  model2,  model3 ]
colors = [ (255,0,0), (255,0,0), (255,0,0)]
times = np.zeros((3,FRAMES_AVG))
fr_count  = 16

def find_faces(img):
    global fr_count

    fr_count += 1
    img_cv = img[0].copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    for k in range(len(models)):
        model = models[k]
        col = colors[k]

        times[k,fr_count%FRAMES_AVG] = times[k,(fr_count+15)%FRAMES_AVG] - time.perf_counter()
        pr = model(img).numpy()
        times[k,fr_count%FRAMES_AVG] += time.perf_counter()

        continue

        for j in range(len(pr)):
            if(pr[j,0] != 0):
                break
            a = pr[j,3]
            angle = pr[j,4]
            centre_x = int(pr[j,1])
            centre_y = int(pr[j,2])

            p1 = str(int(pr[j,5]*1000)/10)
            #p2 = str(int(sigmoid(faces[j,4])*1000)/10)
            cv2.putText(img_cv, p1, (centre_x - 25,centre_y - 25), font, 1, col)            
            #cv2.putText(img_cv, p2, (centre_x - 25,centre_y), font, 1, (0, 255, 0))            
            for j in range(4):
                p1s = (a,-a)
                p2s = (a,+a)

                p1d = ( int(cos(angle)*p1s[0]+sin(angle)*p1s[1]+centre_x), int(-sin(angle)*p1s[0]+cos(angle)*p1s[1]+centre_y))
                p2d = ( int(cos(angle)*p2s[0]+sin(angle)*p2s[1]+centre_x), int(-sin(angle)*p2s[0]+cos(angle)*p2s[1]+centre_y))

                cv2.line(img_cv, p1d, p2d, col, 1)
                angle += math.pi/2

    return img_cv

import time

#cap = cv2.VideoCapture('media/77677667.mp4')
#cap = cv2.VideoCapture('media/video.mp4')
cap = cv2.VideoCapture(0)

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

    img = np.stack( [frame] * MULTIPLE, axis = 0)
    total_time = -time.perf_counter()
    frame = find_faces(img)
    total_time += time.perf_counter()
    total_frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

    tcur = (times[:,fr_count%FRAMES_AVG] - times[:,(fr_count+1)%FRAMES_AVG])*1000/MULTIPLE/FRAMES_AVG
    print(fr_count, "%.2fms %.2fms %.2fms" % (tcur[0], tcur[1], tcur[2]), end = '\r')
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break    