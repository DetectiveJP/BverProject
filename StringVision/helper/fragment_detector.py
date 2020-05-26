import cv2


def find_fragment(img):
    img_frag = img.copy()

    contours, hierarchy = cv2.findContours(img_frag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #  convexity defect
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if type(defects) != type(None):  # avoid crashing.   (BUG not found)
        cnts = 0
        for i in range(defects.shape[0]):
            s, e, f, distance = defects[i][0]
            far = tuple(cnt[f][0])
            #print(distance)
            far_point = distance
            if far_point > 3000:
                cnts += 1
                cv2.circle(img_frag, far, 15, [211, 140, 0], -1)
        return cnts, img_frag
