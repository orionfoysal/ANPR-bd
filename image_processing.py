# import numpy as np
# import os
# import PIL
# from PIL import ImageEnhance

# #from PIL import Image, ImageEnhance
# from sklearn.externals import joblib
# from skimage.io import imread
# from skimage import measure, util
# from skimage.transform import resize
# from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata
# from sklearn import neighbors
# from skimage.measure import regionprops
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from datetime import datetime
# import cv2
# from skimage.viewer import ImageViewer

# sample_image_path = "./Photos/8.jpg"

# image = imread(sample_image_path, as_grey=True)

# img = PIL.Image.open(sample_image_path)
# converter = ImageEnhance.Color(img)
# gray_image = converter.enhance(1)

# gray_image = cv2.cvtColor(np.array(gray_image), cv2.COLOR_RGB2BGR)
# gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

# threshold_value = threshold_otsu(gray_image)

# binary_image = gray_image > threshold_value #+ 100



# fig, (ax1) = plt.subplots(1)
# ax1.imshow(gray_image, cmap="gray")

# plt.show()
# # viewer = ImageViewer(img2)
# # viewer.show()



# # gray_image = image
# # threshold_value = threshold_otsu(gray_image)
# # binary_image = gray_image > threshold_value

# # fig, (ax1, ax2) = plt.subplots(1,2)
# # ax1.imshow(gray_image, cmap="gray")
# # ax2.imshow(binary_image, cmap="gray")
# # plt.show()



import cv2
import matplotlib.pyplot as plt
#-----Reading the image-----------------------------------------------------
img = cv2.imread('./Photos/8.0.jpg', 1)


plt.figure()
plt.imshow(img)
plt.show()
# cv2.imshow("img",img) 

# #-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
plt.imshow(lab)
plt.show()
# cv2.imshow("lab",lab)

# #-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
plt.imshow(l)
plt.show()
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
# cv2.imshow('l_channel', l)
# cv2.imshow('a_channel', a)
# cv2.imshow('b_channel', b)

# #-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
plt.imshow(cl)
plt.show()
# cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
plt.imshow(limg)
plt.show()
# cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
plt.imshow(final)
plt.show()
# cv2.imshow('final', final)

# _____END_____#

plt.show()