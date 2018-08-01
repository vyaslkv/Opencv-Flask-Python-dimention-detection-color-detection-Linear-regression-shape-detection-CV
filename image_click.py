import cv2

def click_image():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cam.set(cv2.CAP_PROP_FOCUS, 0.35)
    # cam.set(cv2.CAP_PROP_FOCUS, 0.098)
    cam.set(cv2.CAP_PROP_FOCUS, 0.30)

    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # cam.set(cv2.CAP_PROP_EXPOSURE, 0.125)
    cam.set(cv2.CAP_PROP_EXPOSURE, 0.0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):  # wait for 's' key to save and exit
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path1, img_name), frame)
            print("{} written!".format(img_name))
        img_counter += 1
    cam.release()
    cv2.destroyAllWindows()