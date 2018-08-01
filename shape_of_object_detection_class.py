import cv2


class ShapeDetection:

    def __init__(self, oblong,ellip,oval,circle,capsule,test):
        self.oblong = oblong
        self.ellip = ellip
        self.oval = oval
        self.circle = circle
        self.capsule = capsule
        self.test = test
        self.image_pre_processing()

    def image_pre_processing(self):
        """

        :return:
        """
        oblong_gray = cv2.cvtColor(self.oblong, cv2.COLOR_BGR2GRAY)
        ellip_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
        oval_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
        circle_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
        capsule_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(self.test, cv2.COLOR_BGR2GRAY)

        ret, thresh_oblong = cv2.threshold(oblong_gray, 45, 255, 0)
        ret, thresh_oval = cv2.threshold(oval_gray, 45, 255, 0)
        ret, thresh_ellip = cv2.threshold(ellip_gray, 45, 255, 0)
        ret, thresh_circle = cv2.threshold(circle_gray, 45, 255, 0)
        ret, thresh_capsule = cv2.threshold(capsule_gray, 35, 255, 0)
        ret, thresh_test = cv2.threshold(test_gray, 45, 255, 0)

        im, contours, hierarchy = cv2.findContours(thresh_oblong, 2, 1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                self.cnt1_oblong = c

        im, contours, hierarchy = cv2.findContours(thresh_ellip, 2, 1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                self.cnt1_ellip = c

        im, contours, hierarchy = cv2.findContours(thresh_oval, 2, 1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                self.cnt1_oval = c

        im, contours, hierarchy = cv2.findContours(thresh_capsule, 2, 1)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                self.cnt1_capsule = c

        im, contours, hierarchy = cv2.findContours(thresh_circle, 2, 1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                self.cnt2_circle = c

        im, contours, hierarchy = cv2.findContours(thresh_test, 2, 1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 50:
                self.cnt3_test = c

    def shape_matching(self):
        """

        :return:
        """
        cnt_dict = {}
        cnt_dict['oblong'] = self.cnt1_oblong
        cnt_dict['oval'] = self.cnt1_oval
        cnt_dict['ellip'] = self.cnt1_ellip
        cnt_dict['circle'] = self.cnt2_circle
        cnt_dict['capsule'] = self.cnt1_capsule
        score_dict = {}
        for shape, values in list(cnt_dict.items()):
            score = cv2.matchShapes(values, self.cnt3_test, 1, 0.0)
            score_dict[shape] = score
        print("score dict", score_dict)
        shape = min(score_dict, key=score_dict.get)

        return shape

