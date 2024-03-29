import cv2
import numpy as np

image = cv2.imread("151.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('second', thresh)
# cv2.waitKey(0)

kernel = np.ones((5, 40), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilated', img_dilation)
# cv2.waitKey(0)

ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi = image[y:y + h, x:x + w]
    # cv2.imshow('segment no: ' + str(i), roi)
    cv2.rectangle(image, (x, y), (x+w, y+h), (90, 0, 255), 2)
    cv2.waitKey(0)

cv2.imwrite('result/'+str(i)+'.png',image)
cv2.imshow('Make Area', image)
cv2.waitKey(0)
