from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import webcolors
from PIL import Image
import cv2
import numpy as np


class ColorDetection:

    def __init__(self,pasted_crops_image):
        self.pasted_crops_image = pasted_crops_image

    def centroid_histogram(self,clt):
        """

        :param clt:
        :return:
        """
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

    def plot_colors(self,hist, centroids):
        """

        :param hist:
        :param centroids:
        :return:
        """
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        for (percent, color) in zip(hist, centroids):
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
            startX = endX
        return bar

    def closest_colour(self,requested_colour):
        """

        :param requested_colour:
        :return:
        """
        min_colours = {}
        webcolors.css3_hex_to_names['#000080'] = 'maroon'
        webcolors.css3_hex_to_names['#ff69b4'] = 'hotpink'
        webcolors.css3_hex_to_names['#ff1493'] = 'deeppink'
        webcolors.css3_hex_to_names['#c71585'] = 'mediumvoiletred'
        webcolors.css3_hex_to_names['#660033'] = 'darkpink'
        webcolors.css3_hex_to_names['#7d3759'] = 'darkpink'
        webcolors.css3_hex_to_names['#ffff00'] = 'yellow'
        webcolors.css3_hex_to_names['#cdcd00'] = 'yellow'
        webcolors.css3_hex_to_names['#b8b800'] = 'yellow'
        webcolors.css3_hex_to_names['#a4a400'] = 'yellow'
        webcolors.css3_hex_to_names['#8f8f00'] = 'yellow'
        webcolors.css3_hex_to_names['#7b7b00'] = 'yellow'
        webcolors.css3_hex_to_names['#999900'] = 'yellow'
        webcolors.css3_hex_to_names['#666600'] = 'darkyellow'
        webcolors.css3_hex_to_names['#333300'] = 'darkyellow'
        webcolors.css3_hex_to_names['#969623'] = 'darkyellow'
        webcolors.css3_hex_to_names['#ff4500'] = 'orangered'
        webcolors.css3_hex_to_names['#ff8c00'] = 'darkorange'
        webcolors.css3_hex_to_names['#ff7f50'] = 'coral'
        webcolors.css3_hex_to_names['#994c00'] = 'darkorange'
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name

        return min_colours[min(min_colours.keys())]

    def get_colour_name(self,requested_colour):
        """

        :param requested_colour:
        :return:
        """
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = self.closest_colour(requested_colour)
            actual_name = None

        return actual_name, closest_name

    def preprocessing_and_nearest_color_from_dict(self):
        """

        :return:
        """
        image = self.pasted_crops_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=3)
        clt.fit(image)
        hist = self.centroid_histogram(clt)
        bar = self.plot_colors(hist, clt.cluster_centers_)
        bar = Image.fromarray(bar)
        size = bar.size
        colors_list = bar.getcolors(size[0] * size[1])
        rgb_count_dict = {}
        for count, rgb in colors_list:
            rgb_count_dict[rgb] = count
        key_to_delete = max(rgb_count_dict, key=lambda k: rgb_count_dict[k])
        del rgb_count_dict[key_to_delete]
        key_of_max_count = max(rgb_count_dict, key=lambda k: rgb_count_dict[k])
        max_count = rgb_count_dict[key_of_max_count]
        key_of_min_count = min(rgb_count_dict, key=lambda k: rgb_count_dict[k])
        min_count = rgb_count_dict[key_of_min_count]
        threshold_value = ((rgb_count_dict[key_of_max_count]) / 2) + ((rgb_count_dict[key_of_max_count]) / 10)
        for rgb, count in list(rgb_count_dict.items()):
            if count < threshold_value:
                rgb_count_dict.pop(rgb)
        key_of_max_count = max(rgb_count_dict, key=lambda k: rgb_count_dict[k])
        max_count = rgb_count_dict[key_of_max_count]
        key_of_min_count = min(rgb_count_dict, key=lambda k: rgb_count_dict[k])
        min_count = rgb_count_dict[key_of_min_count]
        if max_count - min_count > 1000:
            rgb_count_dict.pop(key_of_min_count)

        print("colors dict", rgb_count_dict)
        rgb_list = []
        for rgb in list(rgb_count_dict.keys()):
            rgb_list.append(rgb)


        final_detected_colors = []
        for rgb in rgb_list:
            requested_colour = rgb
            actual_name, closest_name = self.get_colour_name(requested_colour)
            print ("Actual colour name:", actual_name, ", closest colour name:", closest_name)
            final_detected_colors.append(closest_name)

        return final_detected_colors
