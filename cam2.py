import cv2
camera = "/base/soc/i2c0mux/i2c@1/imx219@10"
pipeline = "libcamerasrc camera-name=%s ! video/x-raw,width=640,height=480,framerate=10/1,format=RGBx ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! appsink" % (camera)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
