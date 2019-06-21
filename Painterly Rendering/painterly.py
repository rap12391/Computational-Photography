import sys
import os
import cv2
import math
import numpy as np
import pandas as pd
from random import shuffle

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def paint(s_img):
    canvas = np.ones(s_img.shape)*550
    for radius in radii:
        ref_img = cv2.GaussianBlur(s_img,(0,0),fs*radius,fs*radius)
        out_img = paintLayer(canvas,ref_img, radius)
    return out_img

def paintLayer(canvas, ref_img, radius):
    S = []
    step = int(fg*radius)
    if step//2<=0:
        h_step = 1
    else:
        h_step=step//2
    diff_img = img_diff(canvas,ref_img)
    for x in range(0,canvas.shape[1],step):
        for y in range(0,canvas.shape[0],step):
            x_min = max(0, x-h_step)
            y_min = max(0, y-h_step)
            x_max = min(canvas.shape[1]-1, x+h_step)
            y_max = min(canvas.shape[0]-1, y+h_step)
            M = diff_img[y_min:y_max+1, x_min:x_max+1]
            area_error = M.sum()/((x_max-x_min)*(y_max-y_min))
            if area_error>threshold:
                x1,y1 = np.unravel_index(M.argmax(),M.shape)
                x0 = x1%(h_step)+x_max-h_step
                y0 = y1%(h_step)+y_max-h_step
                s = makeSplineStroke(x0, y0, radius, ref_img, canvas)
                S.append(s)

    shuffle(S)
    for stroke in S:
        stroke = np.array(stroke)
        color = ref_img[stroke[0][1],stroke[0][0],:].astype(int).tolist()
        color.append(a)
        cv2.polylines(canvas,[stroke],False,color, radius)
    return canvas

def makeSplineStroke(x0,y0,R,refImage, canvas):
    strokeColor = refImage[y0,x0,:].astype(int).tolist()
    K = [[x0, y0]]
    x,y = x0, y0
    lastdx, lastdy = 0,0
    gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
    # ksize = int(R*0.7)
    # if ksize<3:
    #     ksize = 3
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=5)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=5)

    for i in range(1, maxStrokeLength+1):
        if i>minStrokeLength and abs(pixel_diff(refImage[y,x,:],canvas[y,x,:])) < abs(pixel_diff(refImage[y,x,:],strokeColor)):
            return K

        gx = sobelX[y,x]
        gy = sobelY[y,x]
        mag = math.sqrt(gx**2 + gy**2)
        if mag == 0:
            return K

        dx, dy = -gy, gx
        if lastdx*dx + lastdy*dy < 0:
            dx, dy = -dx, -dy
        dx = (fc*dx + (1-fc)*lastdx)/math.sqrt(dx**2 + dy**2)
        dy = (fc*dy + (1-fc)*lastdy)/math.sqrt(dx**2 + dy**2)
        x,y = int(x+R*dx), int(y+R*dy)

        if x>canvas.shape[1]-1 or y>canvas.shape[0]-1 or x<0 or y<0:
            break

        lastdx, lastdy = dx, dy
        K.append([x,y])
    return K

def img_diff(canvas, ref_img):
    diff = canvas - 550 - ref_img
    return np.sqrt(np.sum(diff**2, axis=-1))

def pixel_diff(a,b):
    diff = np.sqrt((int(a[0])-int(b[0]))**2 + (int(a[1])-int(b[1]))**2 + (int(a[2])-int(b[2]))**2)
    return diff


if __name__ == "__main__":
    imgs = load_images_from_folder("images")
    i=1
    for img in imgs:

        name = ("%d_impressionist_img.jpg" %(i))
        threshold = 100
        maxStrokeLength = 17
        minStrokeLength = 5
        fc = 1
        fs = 0.5
        a = 1
        fg = 1
        radii = [8,4,2]
        out = paint(img)
        cv2.imwrite(name, out)

        name = ("%d_expressionist_img.jpg" %(i))
        threshold = 50
        maxStrokeLength = 17
        minStrokeLength = 11
        fc = 0.25
        fs = 0.5
        a = 0.7
        fg = 1
        radii = [8,4,2]
        out = paint(img)
        cv2.imwrite(name, out)

        name = ("%d_colorist_img.jpg" %(i))
        threshold = 200
        maxStrokeLength = 17
        minStrokeLength = 5
        fc = 1
        fs = 0.5
        a = 0.5
        fg = 1
        radii = [8,4,2]
        out = paint(img)
        cv2.imwrite(name, out)

        name = ("%d_pointillist_img.jpg" %(i))
        threshold = 100
        maxStrokeLength = 1
        minStrokeLength = 1
        fc = 1
        fs = 0.5
        a = 1
        fg = 0.5
        radii = [4,2]
        out = paint(img)
        cv2.imwrite(name, out)

        i+=1
