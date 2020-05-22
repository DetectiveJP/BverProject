import cv2


# load the image, convert it to grayscale, and blur it slightly
def load_images():

    filename = r"C:\Users\dnns.hrrn\Dropbox\bver_Projekt\Bilder\Mit_IR-Belechtung_Diffusor\Produkt\Image__2020-05" \
               r"-15__11-20-25.bmp "
    print(filename)

    img_src = filename

    img_out = cv2.imread(img_src, 0)

    if img_out is None:
        print("WARNING: No images loaded!")

    return img_out
