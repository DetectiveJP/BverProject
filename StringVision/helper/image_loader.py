import os
import cv2


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
