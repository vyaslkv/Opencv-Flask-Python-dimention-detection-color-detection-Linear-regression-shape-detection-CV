from flask import Flask, request,session, flash, url_for, redirect, render_template, send_from_directory
from flask_classful import FlaskView, route
from jinja2 import Template
import cv2
import io
import numpy as np
import os
import sys
from PIL import Image
import json
from skimage.util import img_as_ubyte
import pickle
from shape_of_object_detection_class import *
from object_size_detection_class import *
from color_detection_class import *
from crops_handling_pasting import *
from maskrcnn_data_collection import *
from image_click import *
from send_data_to_database import *

app=Flask(__name__)
cwd = os.getcwd()
path = cwd+'/static'
path1 = path +'/css/images'

class MasterClass(FlaskView):
    # route_base = '/'

    def __init__(self):

        self.oblong = cv2.imread('1_white.png')
        self.ellip = cv2.imread('thresh_test.png')
        self.oval = cv2.imread('oval.png')
        self.circle = cv2.imread('2_2_1.png')
        self.capsule = cv2.imread('12.png')
        self.test = cv2.imread('threshcrops/1.png')


    @route('/', methods=['GET', 'POST'])
    def index(self):
        """
        renders the home page
        :return:
        """
        return render_template('sq_query_v03.html')

    @route('/new', methods=['GET', 'POST'])
    def data_from_frontend_and_function_calling(self):
        """

        :return:
        """
        if request.method == 'POST':
            """
            receiving the images from the front end
            """
            if request.files.get('picture1', ''):
                # accepting images from front end and showing them after conversion
                image1 = request.files.get('picture1', '')

                in_memory_file = io.BytesIO()
                image1.save(in_memory_file)
                data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
                color_image_flag = 1
                img_bl = cv2.imdecode(data, color_image_flag)
                # img_bl =  img_bl[120:600 ,320:950]
                image3 = request.files.get('picture2', '')
                in_memory_file = io.BytesIO()
                image3.save(in_memory_file)
                data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
                color_image_flag = 1
                img_fl = cv2.imdecode(data, color_image_flag)
                # img_fl = img_fl[120:600 ,320:950]
                image4 = request.files.get('picture3', '')
                in_memory_file = io.BytesIO()
                image4.save(in_memory_file)
                data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
                color_image_flag = 1
                img_bottom_size = cv2.imdecode(data, color_image_flag)


                """
                receiving the drug name form the front end
                """
                pill_name = request.form.get('pillname')
                print("pill name", pill_name)
                f = open('results.txt', 'a')
                f.write("pill name " + str(pill_name))
                f.write('\n')
                f.close()
                B = int(request.form.get('r'))
                G = int(request.form.get('g'))
                R = int(request.form.get('b'))


                tempxf, tempyf, cntlen, pillname = rotate_crops_and_cordinates_for_maskrcnn(img_bl, img_fl, pill_name)
                json_create_for_maskrcnn(tempxf, tempyf, cntlen, pillname)
                pasting_crops_on_background(R, G, B, pillname)

                self.dimention_detection = ObjectSizeMeasurement(img_bottom_size)
                dimA, dimB = self.dimention_detection.bounding_box_mid_points_distance_calculation()
                self.shape_detection = ShapeDetection(self.oblong, self.ellip, self.oval, self.circle, self.capsule,self.test)
                shape = self.shape_detection.shape_matching()
                print("shape", shape)

                filename = 'regression_model.sav'
                loaded_model = pickle.load(open(filename, 'rb'))
                error1 = loaded_model.predict([[dimA]])
                error2 = loaded_model.predict([[dimB]])
                final_height = [dimA] - error1
                final_width = [dimB] - error2
                final_height = ", ".join(repr(e) for e in final_height)
                final_width = ", ".join(repr(e) for e in final_width)
                print("final height", final_height)
                print("final width", final_width)
                hists = os.listdir('static/crops/')

                return render_template('sq_response.html', hists=hists)


    @route('/display', methods=['GET', 'POST'])
    def display(self):
        hists = os.listdir('static/css/images')
        return render_template('sq_query_v03.html', hists=hists)

    @route('/click', methods=['GET', 'POST'])
    def click(self):
        """
        clicks the images of pills
        :return:
        """
        click_image()
        return render_template('home.html')

    @route('/db', methods=['GET', 'POST'])
    def db(self):
        """
        sends the data to database
        :return:
        """
        to_database()
        return render_template('home.html')
# MasterClass.register(app, strict_slashes=False)
if __name__ == '__main__':
    c = MasterClass()
    MasterClass.register(app, route_base='/')
    app.run(port=5000, debug=True)

