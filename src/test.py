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

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
model1 = tf.keras.models.load_model('models/final_large.h5')
model2 = tf.keras.models.load_model('models/1003.h5')
model3 = tf.keras.models.load_model('models/medium.h5')
pad_size = 64
threshold = 0.5

models = [ model1, model2, model3 ]
colors = [ (255,0,0), (0,255,0), (0,0,255)]

pad_size = 256
threshold = 0.5

def find_faces(img, maxwidth, r):
    if len(img.shape) == 2:
        img = np.stack( [img,img,img], axis = -1)

    w = img.shape[1]
    h = img.shape[0]
    scale = 1

    assert(maxwidth%pad_size == 0)

    if w > maxwidth and w > h:
        scale = maxwidth/w
        h = (maxwidth*h + w//2)//w
        w = maxwidth
    elif h > maxwidth:
        scale = maxwidth/h
        w = (maxwidth*w + h//2)//h
        h = maxwidth
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    osth = (pad_size*1000-h)%pad_size
    ostw = (pad_size*1000-w)%pad_size
    if osth != 0 or ostw != 0:
        img = np.pad(img, ((0,osth),(0,ostw),(0,0)))
        h += osth
        w += ostw
    
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    img = img.astype(np.float32)/127.5-1
    img_cv = (((img+1)*127.5)[0]).astype(np.uint8)

    for f in r[1]:
        a = f[2]*scale
        angle = f[3]
        centre_x = f[0]*scale
        centre_y = f[1]*scale

        if a < 12 or a > 64:
            continue
        
        if f[4] == 0:
            color = (0,255,255)
        else:
            color = (255,255,0)

        for j in range(4):
            p1s = (a,-a)
            p2s = (a,+a)

            p1d = ( int(math.cos(angle)*p1s[0]+math.sin(angle)*p1s[1]+centre_x), int(-math.sin(angle)*p1s[0]+math.cos(angle)*p1s[1]+centre_y))
            p2d = ( int(math.cos(angle)*p2s[0]+math.sin(angle)*p2s[1]+centre_x), int(-math.sin(angle)*p2s[0]+math.cos(angle)*p2s[1]+centre_y))

            cv2.line(img_cv, p1d, p2d, color, 3)
            angle += math.pi/2

    font = cv2.FONT_HERSHEY_SIMPLEX
    def sigmoid(x):
        return 1/(1 + np.exp(-x))


    for model, col in zip(models,colors):
        pr = model(img)

        centers = pr[:,:,:,0:1]
        centers_mp = tf.nn.max_pool2d(centers, ksize=(3, 3), strides=(1, 1), padding="SAME")[:,:,:,0]
        centers = centers[:,:,:,0]
        faces = tf.logical_and(tf.greater(tf.nn.sigmoid(centers), threshold), tf.equal(centers, centers_mp))
        centers = tf.where(faces)
        print(centers.numpy())
        faces = tf.gather_nd(pr, centers)
        num_faces = len(faces)

        
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
            sig = sigmoid(faces[j])*100
            print(j, "%.1fpx %.2f%% %.2f%% %.2f%%"%(a, sig[0], sig[5], sig[6]))

            cv2.putText(img_cv, str(j), (centre_x - 5,centre_y - 5), font, 0.5, col)            
            for j in range(4):
                p1s = (a,-a)
                p2s = (a,+a)

                p1d = ( int(cos(angle)*p1s[0]+sin(angle)*p1s[1]+centre_x), int(-sin(angle)*p1s[0]+cos(angle)*p1s[1]+centre_y))
                p2d = ( int(cos(angle)*p2s[0]+sin(angle)*p2s[1]+centre_x), int(-sin(angle)*p2s[0]+cos(angle)*p2s[1]+centre_y))

                cv2.line(img_cv, p1d, p2d, col, 1)
                angle += math.pi/2

    plt.imshow(np.array(img_cv))
    plt.show()


path = r'y:\FindFace/'
with open(path + 'filtered.txt') as json_file:
    records = json.load(json_file)
print("Loaded " + str(len(records)) + " records")
shuffle(records)

for r in records:
    im = Image.open(path + r[0])
    im = np.array(im)
    find_faces(im,256, r)

