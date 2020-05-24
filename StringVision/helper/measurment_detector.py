import cv2
import numpy as np


def get_center(corner_points):
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

    alpha = np.arctan(g / a)
    angle = alpha

    cell = [corner_1, corner_2, corner_3, corner_4, [x_center, y_center], [angle, 0]]
    return x_center, y_center, angle


def get_width(img, pos, angle):
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

    return cell_width_px


def draw_center(center_x, center_y, alpha, img):
    font = cv2.FONT_HERSHEY_COMPLEX
    x1 = center_x - 10
    x2 = center_x + 10
    y1 = center_y - 10
    y2 = center_y + 10
    line_thickness = 2
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=line_thickness)
    img_out = cv2.line(img, (x1, y2), (x2, y1), (0, 0, 255), thickness=line_thickness)

    alpha
    label = "X: " + str(center_x) + " / Y: " + str(center_y)
    cv2.putText(img_out, label, (center_x + 20, center_y - 5), font, 0.5, (255, 0, 0))
    label = "Alpha: " + str(np.rad2deg(alpha))
    cv2.putText(img_out, label, (center_x + 20, center_y + 15), font, 0.5, (255, 0, 0))
    return img_out



