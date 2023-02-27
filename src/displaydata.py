import tensorflow as tf
import numpy as np
import cv2 as cv2
import math
import os
from . import config

def displayData(ds):
    por_num = 0
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    for rec in ds.take(4):
        ai,bi = rec
        for i in range(64):
            img = ai[i].numpy()
            msk = bi[i].numpy()
            img = cv2.cvtColor((img*127.5 + 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            ind  = np.where(msk[:,:,0] == 0)
            ind =  list(zip(ind[0], ind[1]))
            for find in ind:
                f = msk[find]
                centre_x = find[1]*config.OUTPUT_SCALE 
                centre_y = find[0]*config.OUTPUT_SCALE
                cv2.circle(img, (int(centre_x), int(centre_y)), 3, (255,0,0), 1)
            
            ind  = np.where(msk[:,:,1] == 1)
            ind =  list(zip(ind[0], ind[1]))
            for find in ind:
                f = msk[find]
                
                a =  f[4]
                angle = f[5]

                centre_x = (find[1]+f[2])*config.OUTPUT_SCALE 
                centre_y = (find[0]+f[3])*config.OUTPUT_SCALE

                colors = [ (255,0,0), (0,255,0),(255,0,255),(255,255,0),]
                for j in range(4):
                    p1s = (a,-a)
                    p2s = (a,+a)

                    p1d = ( int(math.cos(angle)*p1s[0]+math.sin(angle)*p1s[1]+centre_x), int(-math.sin(angle)*p1s[0]+math.cos(angle)*p1s[1]+centre_y))
                    p2d = ( int(math.cos(angle)*p2s[0]+math.sin(angle)*p2s[1]+centre_x), int(-math.sin(angle)*p2s[0]+math.cos(angle)*p2s[1]+centre_y))

                    cv2.line(img, p1d, p2d, colors[j], 1)
                    angle += math.pi/2
        

            cv2.imwrite('temp/img'+str(por_num)+'.jpg', img)
            por_num += 1
