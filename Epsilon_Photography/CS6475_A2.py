
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import cv2
import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


monkey_img1 = cv2.imread("monkey1.jpg", cv2.IMREAD_COLOR)
monkey_img1 = cv2.resize(monkey_img1, None, fx=0.5, fy=0.5)

monkey_img2 = cv2.imread("monkey2.jpg", cv2.IMREAD_COLOR)
monkey_img2 = cv2.resize(monkey_img2, None, fx=0.5, fy=0.5)

monkey_img3 = cv2.imread("monkey3.jpg", cv2.IMREAD_COLOR)
monkey_img3 = cv2.resize(monkey_img3, None, fx=0.5, fy=0.5)

monkey_img4 = cv2.imread("monkey4.jpg", cv2.IMREAD_COLOR)
monkey_img4 = cv2.resize(monkey_img4, None, fx=0.5, fy=0.5)

monkey_img5 = cv2.imread("monkey5.jpg", cv2.IMREAD_COLOR)
monkey_img5 = cv2.resize(monkey_img5, None, fx=0.5, fy=0.5)

r, c, d= monkey_img1.shape
#print(r,c,d)

buffer_matrix = 255*np.ones((r, 300, d), dtype=int)
#print(buffer_matrix)

new_img = np.concatenate((monkey_img1, buffer_matrix, monkey_img2, buffer_matrix, monkey_img3, buffer_matrix, monkey_img4, buffer_matrix, monkey_img5), axis=1)

#plt.imshow(new_img);plt.axis("off")

cv2.imwrite("final_artifact.jpg", new_img)


# In[4]:


disk1 = cv2.imread("disk1.jpg", cv2.IMREAD_COLOR)
disk1 = cv2.resize(disk1, None, fx=0.5, fy=0.5)

disk2 = cv2.imread("disk2.jpg", cv2.IMREAD_COLOR)
disk2 = cv2.resize(disk2, None, fx=0.5, fy=0.5)

disk3 = cv2.imread("disk3.jpg", cv2.IMREAD_COLOR)
disk3 = cv2.resize(disk3, None, fx=0.5, fy=0.5)

disk4 = cv2.imread("disk4.jpg", cv2.IMREAD_COLOR)
disk4 = cv2.resize(disk4, None, fx=0.5, fy=0.5)

disk5 = cv2.imread("disk5.jpg", cv2.IMREAD_COLOR)
disk5 = cv2.resize(disk5, None, fx=0.5, fy=0.5)


add1 = cv2.addWeighted(disk1,0.5, disk2,0.5,0)
add2 = cv2.addWeighted(add1,0.66, disk3,0.33,0)
add3 = cv2.addWeighted(add2,0.75, disk4, 0.25,0)
add4 = cv2.addWeighted(add3,0.80, disk5, 0.20,0)

#plt.imshow(add4)

cv2.imwrite("final_artifact2.jpg", add4)

