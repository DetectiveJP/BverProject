import imutils
import numpy as np
import cv2
import sys
from helper.image_prepartion import extract_cell
from helper.image_loader import load_images


# show original image
# cv2.imshow('Original', cv2.resize(gray, (1500, 800)))


#gray = load_images()
flooded = extract_cell(gray)



# Vertical = x, horizontal = y
x1 = 0
x2 = 1200
y1 = 1500
#y2 = 901

cropped = flooded[x1:x2, y1:y1+1].copy()
a = x1
while cropped[a, 0] != 255:
    a = a + 1
b = x2 - 1
while cropped[b, 0] != 255:
    b = b - 1
  
pixelSum = np.sum(cropped[a:b, 0])
print("Sum: " + str(pixelSum))

cell_width_px = b - a
print("Cell width [px]: " + str(cell_width_px))

x_scale_factor = 0.166313559322033

cell_width_mm = cell_width_px * x_scale_factor
print("Cell width [mm]: " + str(cell_width_mm))

#cv2.imwrite('binary.bmp', binary)
cv2.imwrite('cropped.bmp', cropped)

#flooded = cv2.floodFill(eroded, cv2::Point(0,0), Scalar(255));
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Binary', binary)
#lines = cv2.HoughLines(binary,1,np.pi/180,200)
#cv2.imwrite('flooded.bmp',flooded)
#cv2.imwrite('cropped.bmp',cropped)
#edge detection
#edged = cv2.Canny(gray, 50, 500, apertureSize=5)
#edged = cv2.dilate(edged, None, iterations=1)
#edged = cv2.erode(edged, None, iterations=1)
# Write image to file
#cv2.imwrite('edged.bmp',edged)
#show original image
#cv2.imshow('Edges', cv2.resize(edged, (1500, 800)))

cv2.waitKey()
cv2.destroyAllWindows()