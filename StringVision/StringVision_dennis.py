import os
import numpy as np
import cv2

from helper.image_prepartion import extract_cell, find_conturs
from helper.image_loader import load_images, load_image_list
from helper.measurment_detector import get_center, draw_center, get_width

# Vertical = x, horizontal = y
x_scale_factor = 0.166313559322033
image_src_path = r"C:\Users\dnns.hrrn\Dropbox\bver_Projekt\Bilder\Mit_IR-Belechtung_Diffusor\Produkt"


list_image = load_image_list(image_src_path)

for filename in list_image:
    img_orig = load_images(os.path.join(image_src_path, filename))
    img_cell = extract_cell(img_orig)

    cell_contoure, img_show_contoures = find_conturs(img_cell)
    cell_center_x, cell_center_y, cell_angle = get_center(cell_contoure)
    img_show_center = draw_center(cell_center_x, cell_center_y, cell_angle, img_show_contoures)


    print(filename)
    cell_width_px = get_width(img_cell, cell_center_y, cell_angle)
    print("Cell width [px]: " + str(cell_width_px))

    cell_width_mm = cell_width_px * x_scale_factor
    print("Cell width [mm]: " + str(cell_width_mm))

    print("Cell orientation [deg]: " + str(np.rad2deg(cell_angle)) + "\n")

    binary_save_path = os.path.join(image_src_path, 'Binary')
    if os.path.exists(binary_save_path):
        cv2.imwrite(os.path.join(binary_save_path, filename[:-4] + '_binary.bmp'), img_cell)

    result_save_path = os.path.join(image_src_path, 'Result')
    if os.path.exists(result_save_path):
        cv2.imwrite(os.path.join(result_save_path, filename[:-4] + '_result.bmp'), img_show_center)

cv2.waitKey()
cv2.destroyAllWindows()
