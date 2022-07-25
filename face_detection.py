import cv2
import numpy as np

image = cv2.imread('face.png')

# load in classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces
faces = classifier.detectMultiScale(gray_image, 1.1, 4)

# draw bounding boxes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (245, 23, 12), 3)


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()