import numpy as np
import cv2
from skimage import filters



image=cv2.imread(r"C:\Users\W10\Desktop\coin4.jpg")
#image = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
cv2.imshow("Original",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(9,9),0) # blur with standard deviation sigma = 9
cv2.imshow("Blurred",blurred)
cv2.waitKey(0)

#Simple edge detection with canny filter
'''
edge = cv2.Canny(blurred,30,150) 
cv2.imshow("Canny_edged",edge)
cv2.waitKey(0)
'''

#If image need closing operation this codes may use
'''
kernel = np.ones((5,5),np.uint8)
closed=closing = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel,iterations=1)
cv2.imshow("close",closed)
cv2.waitKey(0)
'''

# Different thresholding operations
'''
ret,thresh=cv2.threshold(gray, 200, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
'''

ret, thresh2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("thresh2",thresh2)
cv2.waitKey(0)

(cnts,_) = cv2.findContours(thresh2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Belirlenen Madeni Para : {} ".format(len(cnts)))


coins = image.copy()

for i in range(0,len(cnts)):
 cv2.drawContours(coins,cnts,i,(0,255,0),2)
 cv2.imshow("Coins",coins)
 cv2.waitKey(0)


#Show all contoured coins one by one , if u want to see all coins one by one you can uncomment the code lines
'''
for (i, c) in enumerate(cnts):# we are iterating through our contours
 (x, y, w, h) = cv2.boundingRect(c) # x and y are starting point of rectangle in first contours

 print("Coin #{}".format(i + 1))
 coin = image[y:y + h, x:x + w] #cropping image as same height and width as the contours
 cv2.imshow("Coin", coin)

 mask = np.zeros(image.shape[:2], dtype = "uint8") #initialising mask of same height and width as image
 ((centerX, centerY), radius) = cv2.minEnclosingCircle(c) # extracting the centre of circle and radius of the circle
 cv2.circle(mask, (int(centerX), int(centerY)), int(radius),255, -1) #
 mask = mask[y:y + h, x:x + w]
 cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask)) # finally applying the AND operation on coins using mask
 cv2.waitKey(0)
'''