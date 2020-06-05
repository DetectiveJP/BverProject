import os
import cv2              # OpenCV. (2015). Open Source Computer Vision Library.
import numpy as np      # Oliphant, T. E. (2006). A guide to NumPy (Vol. 1). Trelgol Publishing USA.

from helper import image
from helper import measurment

# Constants
X = 0
Y = 1
ANGLE = 2

# Factors and coefficients
px2mmFactor = 0
mtx = None
dist = None
error = None


def main():
    # Configurations
    file_dir = os.path.dirname(__file__)
    image_src_path_prod = os.path.join(file_dir, "pictures", "product")
    img_src_path_calib = os.path.join(file_dir, "pictures", "calibration")
    chessBoardPattern = (6, 9)
    squareLenght = 26  # Lenght of a square block on the chess board image

    print("Read calibration images with chessboard pattern and calibrate camera ===========\n")
    mtx, dist, repro_error, px2mmFactor = image.calibrate_camera(img_src_path_calib, chessBoardPattern, squareLenght)

    print("Read production images and process them ========================================\n")
    # Load image list
    list_image = image.load_image_list(image_src_path_prod)

    for filename in list_image:
        # Preparation: Load image
        path = os.path.join(image_src_path_prod, filename)
        img_orig = image.load_images(path)

        # Manipulate: Get undistorted image and extract one cell
        img_undi = image.undistorted_image(img_orig, mtx, dist, 0)
        img_cell = image.extract_cell(img_undi)

        # Measurement: Get cell contours, center, width and defect count
        cell_contour, img_result = image.find_contours(img_cell)
        cell_center = measurment.get_cell_center(cell_contour, img_result)
        cell_width_mm, cell_width_px = measurment.get_cell_width(img_cell, cell_center[X], cell_center[ANGLE],
                                                                 px2mmFactor,
                                                                 img_result)
        cell_defect_count = measurment.find_cell_defects(img_cell, cell_contour, img_result)
        measurment.draw_cell_center(cell_center[X], cell_center[Y], cell_center[ANGLE], img_result)
        measurment.draw_legend(filename, img_result)

        # Print result to console
        print("----------------------------------------------------------")
        print("Loaded image for processing: " + filename)
        print("Cell width [px]: " + str(cell_width_mm))
        print("Cell width [mm]: " + str(cell_width_mm))
        print("Cell orientation [deg]: " + str(np.rad2deg(cell_center[ANGLE])))
        print("Cell defect count [n]: " + str(cell_defect_count))
        print("\n")

        # Show result
        cv2.imshow('Processed image', img_result)
        cv2.waitKey(100)

        # Save result
        binary_save_path = os.path.join(image_src_path_prod, 'Binary')
        if os.path.exists(binary_save_path):
            cv2.imwrite(os.path.join(binary_save_path, filename[:-4] + '_binary.bmp'), img_cell)

        result_save_path = os.path.join(image_src_path_prod, 'Result')
        if os.path.exists(result_save_path):
            cv2.imwrite(os.path.join(result_save_path, filename[:-4] + '_result.bmp'), img_result)

        # generate a Protocol
        protocol_save_path = os.path.join(image_src_path_prod, 'Protocol')
        if not os.path.exists(protocol_save_path):
            try:
                os.mkdir(protocol_save_path)
            except OSError:
                print("Creation of the directory %s failed" % path)

        if os.path.exists(protocol_save_path):
            f = open(os.path.join(protocol_save_path, filename[:-4] + '_protocol.txt'), 'w+')
            f.write('Review log' + "\n")
            f.write(("Loaded image: " + filename + "\n"))
            f.write("\n")
            f.write("Cell width [px]: " + str(cell_width_mm) + "\n")
            f.write("Cell width [mm]: " + str(cell_width_mm) + "\n")
            f.write("Cell orientation [deg]: " + str(np.rad2deg(cell_center[ANGLE])) + "\n")
            f.write("Cell defect count [n]: " + str(cell_defect_count) + "\n")
            f.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
