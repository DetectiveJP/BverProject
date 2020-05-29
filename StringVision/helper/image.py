import cv2
import numpy as np
import os

COLOR_CELL = (0, 255, 0)

# load the image, convert it to grayscale, and blur it slightly
def load_images(filename):
    # filename = r"C:\Users\dnns.hrrn\Dropbox\bver_Projekt\Bilder\Mit_IR-Belechtung_Diffusor\Produkt\Image__2020-05-15__11-20-25.bmp "

  #  print(filename)

    img_src = filename

    img_in = cv2.imread(img_src, 0)

    if img_in is None:
        print("WARNING: No images loaded!")

    return img_in


def load_image_list(path):
    directory = path
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".bmp") :
            image_list.append(filename)
        else:
            continue

    return image_list


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


def find_contours(img):
    font = cv2.FONT_HERSHEY_COMPLEX

    img_cont = img.copy()

    img_show = cv2.cvtColor(img_cont, cv2.COLOR_BAYER_GR2RGB)
    # Detecting contours in image.
    contours, _ = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Going through every contours found in the image.
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # draws boundary of contours.
        cv2.drawContours(img_show, [approx], 0, COLOR_CELL, 2)

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
                    cv2.putText(img_show, "Top most: " + string, (x, y),
                                font, 0.5, COLOR_CELL)
                else:
                    # text on remaining co-ordinates.
                    cv2.putText(img_show, string, (x, y),
                                font, 0.5, COLOR_CELL)
            i += 1
        cell_corners = approx

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


def find_chessboard_corners(img, pattern):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # Find the chess board corners, (9x6) = Number of inner corners per a chessboard row and column
    ret, corners = cv2.findChessboardCorners(img, pattern, flags)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

    return ret, objp, corners2


def calibrate_camera(path, pattern, length):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    # objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = load_image_list(path)

    for image in images:
        load_path = os.path.join(path, image)
        img_gray = load_images(load_path)
        print("Image loaded: " + load_path)
        ret, objp, corners2 = find_chessboard_corners(img_gray, pattern)

        if ret:
            # Draw and display the corners
            img_show = cv2.cvtColor(img_gray, cv2.COLOR_BAYER_GR2RGB)
            img_show = cv2.drawChessboardCorners(img_show, pattern, corners2, ret)
            #cv2.imshow('img', img_show)
            #cv2.waitKey(50)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # Calibrate Camera with found parameters
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

    # Calcutate re-projection error to check found parameters
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    error = mean_error / len(objpoints)
    print("Total calibration error: {}".format(error) + "\n")

    # Scale pixel to mm on chessboard
    load_path = os.path.join(path, images[0])
    img_chessboard = load_images(load_path)
    img_undist = undistorted_image(img_chessboard, mtx, dist, 0)
    factor = scale_pixel2mm(img_undist, pattern, length)

    return mtx, dist, error, factor


def scale_pixel2mm(img, pattern, length):
    img_undist = img.copy()
    ret, objp, corners = find_chessboard_corners(img_undist, pattern)

    px2mm_factor = 0
    sum_x = 0
    sum_y = 0

    crns = corners.ravel()

    for i in range(4, 88, 12):
        sum_x += (crns[i + 12] - crns[i])

    for i in range(1, 9, 2):
        sum_y += (crns[i] - crns[i + 2])

    avg_x = sum_x / 7
    avg_y = sum_y / 4
    if abs(avg_x - avg_y) < 5:
        px2mm_factor = length / ((avg_x + avg_y) / 2)
        print("Pixel to mm scale factor: {}".format(px2mm_factor) + "\n")

    return px2mm_factor


def undistorted_image(image, mtx, dist, alpha):
    """
    Quelle: https://www.programcreek.com/python/example/84096/cv2.undistort

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

    # undistort image with matrix and coefficients
    ret = cv2.undistort(image, mtx, dist, None, newcameramtx)

    return ret
