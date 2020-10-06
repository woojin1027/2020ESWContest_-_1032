import numpy as np
import cv2
from matplotlib import pyplot as plt

body_cascade = cv2.CascadeClassifier('../../main/OpenCV/haarcascade/haarcascade_fullbody.xml')
image = cv2.imread('people.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

body = body_cascade.detectMultiScale(gray, 1.01, 2, minSize=(500, 500))
num = 0
for (x,y,w,h) in body :         
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),30)
    num += 1

print("현재 판단되는 인원은 ",num)
image = cv2.resize(image,(500,600))
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()