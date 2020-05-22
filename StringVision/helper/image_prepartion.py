import cv2
import numpy as np


def fill_holes(img, seed, val):
    img_th = img.copy()

    # Copy the thresholded image.
    img_floodfill = img_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels wider than the image.
    h, w = img_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    if img[seed] == val:
        print("WARNING: Filling something you shouldn't")

    cv2.floodFill(img_floodfill, mask, seed, val)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # Combine the two images to get the foreground.
    im_out = img_th | im_floodfill_inv

    return im_out


def remove_boarders(img, seed, val):
    img_border = img.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels wider than the image.
    h, w = img_border.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(img_border, mask, seed, val)

    return img_border


def find_conturs(img):
    font = cv2.FONT_HERSHEY_COMPLEX

    img_cont = img.copy()

    img_show = cv2.cvtColor(img_cont, cv2.COLOR_BAYER_GR2RGB)
    # Detecting contours in image.
    contours, _ = cv2.findContours(img_cont, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    # Going through every contours found in the image.
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv2.drawContours(img_show, [approx], 0, (0, 0, 255), 3)

        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        i = 0

        for j in n:
            if i % 2 == 0:
                x = n[i]
                y = n[i + 1]

                # String containing the co-ordinates.
                string = str(x) + " " + str(y)

                if i == 0:
                    # text on topmost co-ordinate.
                    cv2.putText(img_show, "Arrow tip: " + string, (x, y),
                                font, 0.5, (0, 255, 0))
                else:
                    # text on remaining co-ordinates.
                    cv2.putText(img_show, string, (x, y),
                                font, 0.5, (255, 0, 0))
            i = i + 1

        cell_corners = approx.ravel()

    return cell_corners, img_show


def extract_cell(img):
    img_loaded = img.copy()

    # binary image
    ret, img_binary = cv2.threshold(img_loaded, 100, 255, cv2.THRESH_BINARY_INV)

    img_eroded = cv2.erode(img_binary, None, iterations=1)
    cv2.imwrite('eroded.bmp', img_eroded)

    img_filled = fill_holes(img_eroded, (0, 0), 255)  # (1. horizontal, 2. vertical)
    cv2.imwrite('filled.bmp', img_filled)

    img_border_removed = remove_boarders(img_filled, (0, 600), 0)
    img_border_removed = remove_boarders(img_border_removed, (1600, 600), 0)

    i = 1600
    while img_border_removed[600,  i] != 255:
        i -= 1
        if i == 0:
            break
    if img_border_removed[600 , i] == 255:
        img_border_removed = remove_boarders(img_border_removed, (i, 600), 0)




    cv2.imwrite('borders.bmp', img_border_removed)

    return img_border_removed
