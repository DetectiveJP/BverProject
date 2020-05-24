import cv2
import numpy as np
import os

from helper.image_loader import load_image_list, load_images


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
    while img_border_removed[600, i] != 255:
        i -= 1
        if i == 0:
            break
    if img_border_removed[600, i] == 255:
        img_border_removed = remove_boarders(img_border_removed, (i, 600), 0)

    cv2.imwrite('borders.bmp', img_border_removed)

    return img_border_removed


def calibrate(path, pattern):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = load_image_list(path)

    for image in images:
        img_gray = load_images(os.path.join(path, image))
        img_show = cv2.cvtColor(img_gray, cv2.COLOR_BAYER_GR2RGB)

        # Find the chess board corners, (9x6) = Number of inner corners per a chessboard row and column
        ret, corners = cv2.findChessboardCorners(img_gray, pattern, flags)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img_show = cv2.drawChessboardCorners(img_show, pattern, corners2, ret)
            #cv2.imwrite('chessboard.bmp', img)
            cv2.imshow('img', img_show)
            cv2.waitKey(3000)

        # Calibrate Camera with found parameters
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

        print(str(path) + str(image))

    # Calcutate re-projection error to check found parameters
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    error = mean_error / len(objpoints)
    print("total error: {}".format(error))

    return mtx, dist, error


# Quelle: https://www.programcreek.com/python/example/84096/cv2.undistort
def undistort(image, mtx, dist, alpha):
    """
    image: an image
    alpha = 0: returns undistored image with minimum unwanted pixels (image
                pixels at corners/edges could be missing)
    alpha = 1: retains all image pixels but there will be black to make up
                for warped image correction
    """
    h, w = image.shape[:2]
    # mtx = self.data['camera_matrix']
    # dist = self.data['dist_coeff']
    # Adjust the calibrations matrix
    # alpha=0: returns undistored image with minimum unwanted pixels (image pixels at corners/edges could be missing)
    # alpha=1: retains all image pixels but there will be black to make up for warped image correction
    # returns new cal matrix and an ROI to crop out the black edges
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha)
    # undistort
    ret = cv2.undistort(image, mtx, dist, None, newcameramtx)

    return ret
