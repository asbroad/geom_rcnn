#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from PIL import Image as PImage
from geometry_msgs.msg import PolygonStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
from IPython import embed
import rospkg

class StoreData:

    def __init__(self):
        self.data_dir = rospy.get_param('/store_data/data_dir')
        self.category = rospy.get_param('/store_data/category')
        self.pic_count = rospy.get_param('/store_data/init_idx')
        self.rate = 1/float(rospy.get_param('/store_data/rate'))

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_color', Image, self.img_cb)
        self.patches_sub = rospy.Subscriber('/candidate_regions_depth', PolygonStamped, self.patches_cb)
        # you can read this value off of your sensor from the '/camera/depth_registered/camera_info' topic
        self.P = np.array([[525.0, 0.0, 319.5, 0.0], [0.0, 525.0, 239.5, 0.0], [0.0, 0.0, 1.0, 0.0]])

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

            # crop image based on rectangle, note: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            self.crop_img = self.cv_image[p2_im_y:p1_im_y, p1_im_x:p2_im_x]

            cv2.imshow("Image window", self.cv_image)
            cv2.waitKey(3)

            filename = self.data_dir + self.category + '/' + str(self.pic_count).zfill(4) + '.jpg'
            cv2.imwrite(filename, self.crop_img)
            self.pic_count += 1
            time.sleep(self.rate)


def main():
    store_data = StoreData()
    rospy.init_node('store_data', anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")

if __name__=='__main__':
    main()
