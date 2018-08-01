import cv2
import os
import imutils
import numpy as np
from PIL import Image
from color_detection_class import *

cwd = os.getcwd()
path = cwd+'/static'
path1 = path +'/css/images'

def pasting_crops_on_background(R, G, B, pill_name):
    """

    :param R:
    :param G:
    :param B:
    :param pill_name:
    :return:
    """
    path_th = cwd + '/threshcrops/'
    path_org = cwd + '/orgcrops/'
    thresh_crops = os.listdir(path_th)

    # thresh_crops.sort(key=lambda f: int(filter(str.isdigit, f)))
    org_crops = os.listdir(path_org)
    # org_crops.sort(key=lambda f: int(filter(str.isdigit, f)))
    temp = True
    idx = 0
    temp1 = True
    for i in range(len(thresh_crops)):
        thresh = cv2.imread(path_th + thresh_crops[i])
        org = cv2.imread(path_org + org_crops[i])
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        _, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 200.0:
                img_fg = cv2.bitwise_and(org, org, mask=thresh)
                new_image = img_fg[y:y + h, x:x + w]
                if temp1:
                    nh, nw, nc = new_image.shape
                    nw1 = nw * 3
                    nh1 = nw1 * 1.33
                    temp1 = False
                if temp == True:
                    bg_img = np.zeros((int(nw1), int(nh1), 3), np.uint8)
                    hht, wdt, chnl = bg_img.shape
                    hht1, wdt1, chnl1 = new_image.shape
                    dh = hht - hht1
                    dw = wdt - wdt1 * 2
                    cx = dw / 2
                    cy = dh / 2

                    # bg_img = np.zeros((600,700,3),np.uint8)
                    n_bg_img = Image.fromarray(bg_img)

                    n_new_img = Image.fromarray(new_image)

                    x, y = n_new_img.size
                    n_bg_img.paste(n_new_img, (int(cx), int(cy), int(cx + x), int(cy + y)))
                    pixels = n_bg_img.load()
                    for i1 in range(n_bg_img.size[0]):  # for every pixel:
                        for j1 in range(n_bg_img.size[1]):
                            if pixels[i1, j1] == (0, 0, 0):
                                pixels[i1, j1] = (R, G, B)
                    n_bg_img = np.array(n_bg_img)

                    path = cwd + '/static/crops/'
                    cv2.imwrite(os.path.join(path, str(pill_name) + str(idx) + '.jpg'), n_bg_img)
                else:

                    n_bg_img = Image.fromarray(n_bg_img)

                    n_new_img = Image.fromarray(new_image)

                    x, y = n_new_img.size
                    path = cwd + '/static/crops/'
                    n_bg_img.paste(n_new_img, (int(cx), int(cy), int(cx + x), int(cy + y)))
                    pixels = n_bg_img.load()  # create the pixel map

                    for i1 in range(n_bg_img.size[0]):  # for every pixel:
                        for j1 in range(n_bg_img.size[1]):
                            if pixels[i1, j1] == (0, 0, 0):
                                pixels[i1, j1] = (R, G, B)
                    n_bg_img = np.array(n_bg_img)

                    n_bg_img = cv2.resize(n_bg_img, (288, 216))
                    cv2.imwrite(os.path.join(path, str(pill_name) + str(idx) + '.jpg'), n_bg_img)
                    pasted_image_name = str(pill_name) + str(idx) + '.jpg'
                    pasted_image = cv2.imread(path + pasted_image_name)
                    color_detection = ColorDetection(pasted_image)
                    final_detected_colors = color_detection.preprocessing_and_nearest_color_from_dict()
                    print("final detected colors", final_detected_colors)
                cx = cx + wdt1 + 5
                temp = False

def rotate_crops_and_cordinates_for_maskrcnn(img_bl, img_fg, pillname):

    """
    this function rotates the crops and save them in different folders for further process and also
    saves the contour points for mask rcnn
    :param img_bl:
    :param img_fg:
    :param pillname:
    :return:
    """

    gray = cv2.cvtColor(img_bl, cv2.COLOR_BGR2GRAY)

    retval, threshold_img = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

    threshold_img = cv2.bitwise_not(threshold_img)

    img_fg = cv2.bitwise_and(img_fg, img_fg, mask=threshold_img)

    _, cnt, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    path_th = cwd + '/threshcrops/'
    path_org = cwd + '/orgcrops/'
    idx = 0
    tempx = []
    tempy = []
    tempxf = []
    tempyf = []
    cntlen = 0
    for c in cnt:

        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 800 < area < 100000.0:
            idx += 1
            for item in c:
                for x1, y1 in item:
                    tempx.append(x1)
                    tempy.append(y1)
            tempxf.append(tempx)
            tempyf.append(tempy)
            cntlen = cntlen + 1
            new_image = img_fg[y:y + h, x:x + w]
            path_color = cwd + '/static/color_crops/'
            cv2.imwrite(path_th + str(idx) + '.png', new_image)
            new_image1 = threshold_img[y:y + h, x:x + w]
            if len(c) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                rotation_angle = 180 - angle

                new_image = imutils.rotate_bound(new_image, rotation_angle)
                new_image1 = imutils.rotate_bound(new_image1, rotation_angle)
                cv2.imwrite(path_th + str(idx) + '.png', new_image1)
                cv2.imwrite(path_org + str(idx) + '.png', new_image)

    return tempxf, tempyf, cntlen, pillname