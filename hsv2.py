import cv2
import numpy as np
 
 
def mousehsv(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
        hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
        print ("HSV : " ,hsv)
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
 
image = cv2.imread('/Users/jay/Desktop/Project Fruit/Day1/mango2.JPG')
cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mousehsv)
 
while(1):
    cv2.imshow('mouseRGB',image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()