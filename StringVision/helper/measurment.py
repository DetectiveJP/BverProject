import datetime

import cv2
import numpy as np

COLOR_MEASUREMENT = (238, 0, 238)
COLOR_INFO = (255, 255, 255)
COLOR_CELL = (0, 255, 0)
font = cv2.FONT_HERSHEY_COMPLEX
line_thickness = 2


def get_cell_center(contour, img_result):
    corner_points = contour.ravel()
    # horizontal = x, vertical = y
    # corner_points = [x0 y0 x1 y1 ....
    if (corner_points[1] + corner_points[5]) / 2 > 300:
        corner_1 = [corner_points[2], corner_points[1]]
        corner_2 = [corner_points[4], corner_points[7]]
        corner_3 = [corner_points[8], corner_points[9]]
        corner_4 = [corner_points[10], corner_points[11]]
    else:
        corner_1 = [corner_points[4], corner_points[3]]
        corner_2 = [corner_points[6], corner_points[9]]
        corner_3 = [corner_points[10], corner_points[11]]
        corner_4 = [corner_points[0], corner_points[1]]

    x_center = int((corner_1[0] + corner_2[0] + corner_3[0] + corner_4[0]) / 4)
    y_center = int((corner_1[1] + corner_2[1] + corner_3[1] + corner_4[1]) / 4)

    a = corner_4[0] - corner_1[0]
    g = corner_4[1] - corner_1[1]

    angle = np.arctan(g / a)

    cell_center = (x_center, y_center, angle)

    draw_cell_center(x_center, y_center, angle, img_result)

    return cell_center


def draw_cell_center(center_x, center_y, angle, img_result):
    x1 = center_x - 10
    x2 = center_x + 10
    y1 = center_y - 10
    y2 = center_y + 10

    cv2.line(img_result, (x1, y1), (x2, y2), COLOR_MEASUREMENT, thickness=line_thickness)
    cv2.line(img_result, (x1, y2), (x2, y1), COLOR_MEASUREMENT, thickness=line_thickness)

    label = "X [px]: " + str(center_x) + " / Y [px]: " + str(center_y)
    cv2.putText(img_result, label, (center_x + 20, center_y - 5), font, 0.5, COLOR_MEASUREMENT)
    label = "Angle [deg]: " + str(np.rad2deg(angle))
    cv2.putText(img_result, label, (center_x + 20, center_y + 15), font, 0.5, COLOR_MEASUREMENT)

    return


def get_cell_width(img, pos, angle, factor, img_result):
    w1 = 0
    w2 = img.size - 1

    img_slice = img[w1:w2, pos:pos + 1].copy()
    a = 0
    while a < img_slice.size and img_slice[a, 0] != 255:
        a = a + 1
    b = img_slice.size - 1
    while b > 0 and img_slice[b, 0] != 255:
        b = b - 1

    pixel_sum = np.sum(img_slice[a:b, 0])
    if pixel_sum == 0:
        print("WARNING: Sum = " + str(pixel_sum))

    cell_width_px = (b - a) * np.cos(angle)
    cell_width_mm = cell_width_px * factor

    corr = (b - a) * np.sin(angle) / 2
    label = "Cell width [mm]: {}".format(cell_width_mm)
    cv2.putText(img_result, label, (pos + 10 + int(corr), a + 50), font, 0.5, COLOR_MEASUREMENT)
    cv2.line(img_result, (pos + int(corr), a), (pos - int(corr), b), COLOR_MEASUREMENT, thickness=line_thickness)

    return cell_width_mm, cell_width_px


def find_cell_defects(img, contour, img_result):
    img_gray = img.copy()
    img_cont = img.copy()
    cv2.drawContours(img_cont, [contour], 0, (255, 255, 255), cv2.FILLED)

    # Invert floodfilled image
    img_cont_inv = cv2.bitwise_not(img_cont)

    # Combine the two images to get the foreground.
    img_comb_inv = img_gray | img_cont_inv

    img_comb = cv2.bitwise_not(img_comb_inv)

    img_erod = cv2.erode(img_comb, None, iterations=1)
    cnts, _ = cv2.findContours(img_erod, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    defect_count = 0

    for c in cnts:
        area = cv2.contourArea(c)

        if area > 3:
            pnts = c.ravel()
            label = "Defect area [px^2]: {}".format(area)
            cv2.putText(img_result, label, (pnts[0] + 20, pnts[1]), font, 0.5, (238, 0, 238))
            cv2.drawContours(img_result, [c], -1, COLOR_MEASUREMENT, cv2.FILLED)
            defect_count += 1

    return defect_count


def draw_legend(filename, img_result):
    cv2.arrowedLine(img_result, (30, 30), (130, 30), COLOR_INFO, 1)
    cv2.putText(img_result, "X", (140, 35), font, 0.8, COLOR_INFO)
    cv2.arrowedLine(img_result, (30, 30), (30, 130), COLOR_INFO, 1)
    cv2.putText(img_result, "Y", (20, 160), font, 0.8, COLOR_INFO)

    cv2.rectangle(img_result, (30, 180), (45, 195), COLOR_MEASUREMENT, cv2.FILLED)
    cv2.putText(img_result, "Measurement", (60, 195), font, 0.5, COLOR_MEASUREMENT)

    cv2.rectangle(img_result, (30, 200), (45, 215), COLOR_CELL, cv2.FILLED)
    cv2.putText(img_result, "Cell", (60, 215), font, 0.5, COLOR_CELL)

    info = "Date/Time: " + str(datetime.datetime.now()) + " / File name: " + filename
    cv2.putText(img_result, info, (30, 1200), font, 0.5, COLOR_INFO)

    return
