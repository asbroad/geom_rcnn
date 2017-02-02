#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import Image
from geom_rcnn.msg import Detection
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from keras_cnn import CNN

class RGBObjectDetection:

    def __init__(self):

        self.run_recognition = rospy.get_param('/rgb_object_detection/run_recognition')
        self.model_filename = rospy.get_param('/rgb_object_detection/model_file')
        self.weights_filename = rospy.get_param('/rgb_object_detection/weights_file')
        self.categories_filename = rospy.get_param('/rgb_object_detection/category_file')
        self.verbose = rospy.get_param('/rgb_object_detection/verbose', False)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_color', Image, self.img_cb)
        self.patches_sub = rospy.Subscriber('/candidate_regions_depth', PolygonStamped, self.patches_cb)
        self.detection_pub = rospy.Publisher('/detections', Detection)
        # you can read this value off of your sensor from the '/camera/depth_registered/camera_info' topic
        self.P = np.array([[525.0, 0.0, 319.5, 0.0], [0.0, 525.0, 239.5, 0.0], [0.0, 0.0, 1.0, 0.0]])

        if self.run_recognition:
            self.cnn = CNN('', self.model_filename, self.weights_filename, self.categories_filename, '', 0, 0, self.verbose)
            self.cnn.load_model()

    def img_cb(self, msg):
        try:
          self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)

    def patches_cb(self, msg):
        if hasattr(self, 'cv_image'):
            ul_pc = msg.polygon.points[0]
            lr_pc = msg.polygon.points[1]
            cen_pc = msg.polygon.points[2]

            p1_pc = np.array([[ul_pc.x], [ul_pc.y], [ul_pc.z], [1.0]])
            p2_pc = np.array([[lr_pc.x], [lr_pc.y], [lr_pc.z], [1.0]])

            # transform from xyz (depth) space to xy (rgb) space, http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html
            p1_im = np.dot(self.P,p1_pc)
            p1_im = p1_im/p1_im[2] #scaling
            p2_im = np.dot(self.P,p2_pc)
            p2_im = p2_im/p2_im[2] #scaling

            p1_im_x = int(p1_im[0])
            p1_im_y = int(p1_im[1])
            p2_im_x = int(p2_im[0])
            p2_im_y = int(p2_im[1])

            # x is positive going right, y is positive going down
            width = p2_im_x - p1_im_x
            height = p1_im_y - p2_im_y

            # expand in y direction to account for angle of sensor
            expand_height = 0.4 # TODO: fix hack with transform/trig
            height_add = height * expand_height
            p1_im_y = p1_im_y + int(height_add)
            height = p1_im_y - p2_im_y

            # fix bounding box to be square
            diff = ( abs(width - height) / 2.0)
            if width > height: # update ys
                p1_im_y = int(p1_im_y + diff)
                p2_im_y = int(p2_im_y - diff)
            elif height > width: # update xs
                p1_im_x = int(p1_im_x - diff)
                p2_im_x = int(p2_im_x + diff)

            ## expand total box to create border around object (e.g. expand box 40%)
            expand_box = 0.4
            box_add = (width * expand_box)/2.0
            p1_im_x = int(p1_im_x - box_add)
            p1_im_y = int(p1_im_y + box_add)
            p2_im_x = int(p2_im_x + box_add)
            p2_im_y = int(p2_im_y - box_add)

            # optional : run the recognition portion of the pipeline
            self.pred = ''
            self.pred_val = 0.0
            if self.run_recognition:
                # crop image based on rectangle, note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                self.crop_img = self.cv_image[p2_im_y:p1_im_y, p1_im_x:p2_im_x]

                im = cv2.resize(self.crop_img, (self.cnn.sample_size, self.cnn.sample_size)).astype(np.float32)
                im = im.transpose((2,0,1))
                im = np.expand_dims(im, axis=0)
                im = im.astype('float32')
                im = im/255.0

                pred = self.cnn.model.predict(im)
                self.pred = self.cnn.inv_categories[np.argmax(pred,1)[0]]
                self.pred_val = np.max(pred,1)[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_text = str(self.pred)
                font_size = 1.0
                font_thickness = 2
                cv2.putText(self.cv_image, label_text, (p1_im_x,p1_im_y), font, font_size,(255,255,255), font_thickness)

            #plot expanded bounding box in green
            cv2.rectangle(self.cv_image, (p1_im_x, p1_im_y), (p2_im_x, p2_im_y), (0, 255, 0), thickness=5)

            # publish detection message
            det_msg = Detection()
            det_msg.obj_class = self.pred
            det_msg.point = cen_pc
            self.detection_pub.publish(det_msg)

            # show image window
            cv2.imshow("Image window", self.cv_image)
            cv2.waitKey(3)


def main():
    rgb_object_detection = RGBObjectDetection()
    rospy.init_node('rgb_object_detection', anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")

if __name__=='__main__':
    main()
