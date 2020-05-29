import cv2


def find_fragment(img):
    img_frag = img.copy()
    # finds all contours in image
    contours, hierarchy = cv2.findContours(img_frag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #  makes a hull around the object
    hull = cv2.convexHull(cnt, returnPoints=False)
    # defines the contours next to the defined hull
    # Function-Output: Array with ->[start point, end point, farthest point, approximate distance to farthest point]
    defects = cv2.convexityDefects(cnt, hull)
    cnts = 0
    if type(defects) != type(None):
        for i in range(defects.shape[0]):
            # s for start point, e for end point, f for farthest point,
            # distance for approximate distance to farthest point
            s, e, f, distance = defects[i][0]
            farthest_point = tuple(cnt[f][0])
            far_point = distance
            # Shows only points from the distance 3000
            if far_point > 3000:
                cnts += 1
                # makes a circle at the fragmented position
                img_show = cv2.cvtColor(img_frag, cv2.COLOR_BAYER_GR2RGB)
                cv2.circle(img_show, farthest_point, 12, [0, 0, 255], -1)

                return cnts, img_show
    return cnts, img_frag
