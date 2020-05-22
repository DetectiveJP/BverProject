import cv2
import numpy as np


def fill_holes(img, seed, val):
    im_th = img.copy()

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    if img[0, 0] != val:
        print("WARNING: Filling something you shouldn't")
    cv2.floodFill(im_floodfill, mask, seed, val);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def remove_boarders(img, seed):
    search = img.copy()
    x, y = seed

    if search[x, y] != 0:
        search[x, y] = 0



def extract_cell(img):
    img_loaded = img.copy()

    # binary image
    ret, binary = cv2.threshold(img_loaded, 100, 255, cv2.THRESH_BINARY_INV)

    eroded = cv2.erode(binary, None, iterations=1)
    cv2.imwrite('eroded.bmp', eroded)

    filled = fill_holes(eroded, (0, 0), 255)
    cv2.imwrite('filled.bmp', filled)

    border_removed = fill_holes(filled, (0, 600), 125)
    cv2.imwrite('borders.bmp', border_removed)
    img_out = border_removed

    return img_out
