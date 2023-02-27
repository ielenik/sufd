import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from numpy import matrix
from numpy import linalg
import math


import json
import os
from tqdm import tqdm
from random import shuffle

def load_rkk(path, records):
    with open(path + '/info.txt') as addf:
        for l in addf:
            bt = l.split(";")
            fname = bt[0].split('/')[1]
            ang = float(bt[4])
            if ang > math.pi:
                ang -= 2*math.pi
            if os.path.isfile(path+"/mask/"+fname):
                r = []
                r.append(path+"/mask/"+fname)
                r.append([[ float(bt[1]), float(bt[2]), float(bt[3]), ang, 1, 0, 1, -1]])
                r.append([ 1024, 640 ])
                records.append(r)
            if os.path.isfile(path+"/nomask/"+fname):
                r = []
                r.append(path+"/nomask/"+fname)
                r.append([[ float(bt[1]), float(bt[2]), float(bt[3]), ang, 1, 0, 0, 0]])
                r.append([ 1024, 640 ])
                records.append(r)
            if os.path.isfile(path+"/halfmask/"+fname):
                r = []
                r.append(path+"/halfmask/"+fname)
                r.append([[ float(bt[1]), float(bt[2]), float(bt[3]), ang, 1, 0, 1, 1]])
                r.append([ 1024, 640 ])
                records.append(r)


def load_dataset(path):
    records = []
    with open(path + '/filtered_width.txt') as json_file:
         records = json.load(json_file)

    count_cat = [[0,0],[0,0],[0,0]]
    for r in tqdm(records):
        for f in r[1]:
            if 'mask' in r[0]:
                f[5] = -1
            elif 'Spoof' in r[0]:
                f[7] = -1
                if 'live' in r[0]:
                    f[6] = 0
                else:
                    f[6] = -1
            else:
                f[5] = f[6] = f[7] = -1
            
            for i in range(3):
                if(f[5+i] == 0):
                    count_cat[i][0] += 1
                elif(f[5+i] == 1):
                    count_cat[i][1] += 1
        r[0] = path + '\\' + r[0]

    print(count_cat)

    load_rkk(path + '/dataset/spoof', records)
    load_rkk(path + '/dataset/RKK', records)

    shuffle(records)
    print("Loaded " + str(len(records)) + " records total")
    return records


BATCH_SIZE = 512
tilesize_full = 128
anchors_scale  = 16
tilesize_small = tilesize_full//anchors_scale
part_for_validation = 200


def getFaceDatasets():
    path = 'c:\\databases\\findface'
    records = load_dataset(path)


    def prepare_image_transform(ind):
        filename = records[ind][0]
        w,h = records[ind][2]
        scalefit = tilesize_full/max(w,h)
        if scalefit > 1:
            scalefit = 1
        scale = np.random.randint(30,200)/100.*scalefit

        ranshx = max((w*scale-tilesize_full)/2, 64)
        ranshy = max((h*scale-tilesize_full)/2, 64)

        angle = math.radians(np.random.randint(-60,60))
        shx = np.random.randint(-ranshx,ranshx)
        shy = np.random.randint(-ranshy,ranshy)

        if ind < len(records)//part_for_validation:
            angle = 0
            shx = 0
            shy = 0
            scale = tilesize_full/max(w,h)
            if scale > 1:
                scale = 1

        tr = matrix([[scale*math.cos(angle),scale*math.sin(angle),tilesize_full/2-scale*(w*math.cos(angle)+h*math.sin(angle))/2 + shx],
                    [scale*-math.sin(angle),scale*math.cos(angle),tilesize_full/2-scale*(-w*math.sin(angle)+h*math.cos(angle))/2 + shy],
                    [0,0,1]])
        Tinv = linalg.inv(tr)
        Tinvtuple = np.array([Tinv[0,0],Tinv[0,1], Tinv[0,2], Tinv[1,0],Tinv[1,1],Tinv[1,2],0,0]).astype(np.float32)

        mask = np.zeros((tilesize_small,tilesize_small,8)).astype(np.float32)
        mask[:,:,0] = 1

        for f in records[ind][1]:
            rx = (tr[0,0]*f[0]+tr[0,1]*f[1] + tr[0,2])/anchors_scale
            ry = (tr[1,0]*f[0]+tr[1,1]*f[1] + tr[1,2])/anchors_scale

            x = int(rx)
            y = int(ry)

            s = f[2]*scale
            a = f[3]+angle

            ms = int(round(s/4/anchors_scale))
            sx,ex,sy,ey = x-ms,x+ms+2,y-ms,y+ms+2
            
            if sx < 0: sx = 0
            if sy < 0: sy = 0
            if ex > tilesize_small: ex = tilesize_small
            if ey > tilesize_small: ey = tilesize_small
            
            if sx >= ex or sy >= ey:
                continue

            mask[sy:ey,sx:ex,0] = 0

            if s < 12 or s > 64:
                continue
            if f[4] < 0.5:
                continue

            if x < 0 or x >= tilesize_small or y < 0 or y>= tilesize_small:
                continue
                
            for i in range(x,x+2):
                if i < 0 or i >= tilesize_small:
                    continue
                for j in range(y,y+2):
                    if j < 0 or j>=tilesize_small:
                        continue
                    mask[j,i,1] = 1
                    mask[j,i,2] = (rx - i)
                    mask[j,i,3] = (ry - j)
                    mask[j,i,4] = s
                    mask[j,i,5] = a
                    mask[j,i,6] = f[5]
                    mask[j,i,7] = f[6]
        return filename, Tinvtuple, mask


    def read_image(filename, tr, mask):
        image_file = tf.io.read_file(filename, name='read_file')
        image = tf.image.decode_jpeg(image_file, channels=3)
        tr.set_shape((8))
        mask.set_shape((tilesize_small,tilesize_small,8))
        image = tfa.image.transform(image,tr, output_shape=[tilesize_full,tilesize_full], interpolation = 'nearest')
        image = tf.cast(image,tf.float32)/127.5-1
        return image, mask


    tf_prepare_image_transform = lambda index: tf.numpy_function(prepare_image_transform, [index], [tf.string,tf.float32,tf.float32])

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    train = tf.data.Dataset.from_tensor_slices( tf.range(len(records)//part_for_validation,len(records)) )
    train = train.map(tf_prepare_image_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.apply(tf.data.experimental.ignore_errors())
    train = train.repeat()
    train = train.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    train = train.with_options(options)

    valid = tf.data.Dataset.from_tensor_slices( tf.range(0, len(records)//part_for_validation) )
    valid = valid.map(tf_prepare_image_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid = valid.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid = valid.apply(tf.data.experimental.ignore_errors())
    valid = valid.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    valid = valid.with_options(options)

    return train, valid
