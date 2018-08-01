from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


class ObjectSizeMeasurement:

    def __init__(self,bottom_light_image_with_reference_object):
        """
        initilization of the contants and variables
        :param bottom_light_image_with_reference_object:
        """
        self.bottom_light_image_with_reference_object = bottom_light_image_with_reference_object
        self.width_of_refernce_object = 25
        self.dimA = None
        self.dimB = None
        self.pre_processing_of_image()
    def midpoint(self,ptA, ptB):
        """
        this will calculate the mid points of the bounding lines
        :param ptA:
        :param ptB:
        :return: mid point
        """
        return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

    def pre_processing_of_image(self):
        """
        this will do the preprossing of the image
        :return: edged image
        """

        self.image = self.bottom_light_image_with_reference_object
        # image = image[100:620,300:900]
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        return edged

    def bounding_box_mid_points_distance_calculation(self):
        """
        this method will calculate the distance between the midpoints of bounding lines
        :return: height and width
        """

        cnts = cv2.findContours(self.pre_processing_of_image().copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        for c in cnts:

            if cv2.contourArea(c) < 100:
                continue
            orig = self.image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)

            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric = dB / self.width_of_refernce_object
            self.dimA = dA / pixelsPerMetric
            self.dimB = dB / pixelsPerMetric

        return self.dimA, self.dimB



