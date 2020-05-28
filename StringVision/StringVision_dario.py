import os
import cv2

from helper.fragment_detector import find_fragment
from helper.image_prepartion import extract_cell
from helper.image_loader import load_images, load_image_list


# Vertical = x, horizontal = y
x_scale_factor = 0.166313559322033
image_src_path = r"C:\Users\dario\OneDrive\FHNW\Semester6\bver\Vorlesung\Projekte\Bilder"

list_image = load_image_list(image_src_path)

for filename in list_image:
    img_orig = load_images(os.path.join(image_src_path, filename))
    img_cell = extract_cell(img_orig)
    cnts, img_fragment = find_fragment(img_cell)

    print(filename)
    print("Counted Fragments: " + str(cnts))

    fragment_save_path = os.path.join(image_src_path, 'Fragment')
    if os.path.exists(fragment_save_path):
        cv2.imwrite(os.path.join(fragment_save_path, filename[:-4] + '_fragment.bmp'), img_fragment)

cv2.waitKey()
cv2.destroyAllWindows()
