import cv2
from matplotlib import pyplot as plt 
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "sample picture", "13.jpg")
image = cv2.imread(image_path)[:,:,::-1]

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(2)
plt.subplot(2,1,1)
plt.imshow(image)


plt.figure(2)
plt.subplot(2,1,2)
plt.imshow(img_gray)
plt.show()

