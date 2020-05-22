import imutils
import numpy as np
import cv2
import sys

# load the image, convert it to grayscale, and blur it slightly
imgsrc = r"C:\Users\dnns.hrrn\Dropbox\bver_Projekt\Bilder\Mit_IR-Belechtung_Diffusor\Produkt\Image__2020-05-15__11-20-25.bmp"

gray = cv2.imread(imgsrc, 0)
if gray is None:
    sys.exit('Failed to load the image')

# show original image
# cv2.imshow('Original', cv2.resize(gray, (1500, 800)))

# binary image
ret, binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
eroded = cv2.erode(binary, None, iterations=1)


def fill_holes(img):
    im_th = img.copy()

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    if img[0,0] != 0:
        print("WARNING: Filling something you shouldn't")
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


flooded_inv = fill_holes(eroded)
flooded = cv2.bitwise_not(flooded_inv)

# Vertical = x, horizontal = y
x1 = 0
x2 = 1200
y1 = 1500
#y2 = 901

cropped = flooded[x1:x2, y1:y1+1].copy()
a = x1
while cropped[a, 0] != 0:
    a = a + 1
b = x2 - 1
while cropped[b, 0] != 0:
    b = b - 1
  
pixelSum = np.sum(cropped[a:b, 0])
print("Sum: " + str(pixelSum))

cell_width_px = b - a
print("Cell width [px]: " + str(cell_width_px))

x_scale_factor = 0.166313559322033

cell_width_mm = cell_width_px * x_scale_factor
print("Cell width [mm]: " + str(cell_width_mm))

cv2.imwrite('binary.bmp', binary)
cv2.imwrite('flooded.bmp', flooded)
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