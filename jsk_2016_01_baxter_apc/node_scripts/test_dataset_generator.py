#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import os
import message_filters
import rospy
from sensor_msgs.msg import Image


class TestDataSave(ConnectionBasedTransport):

    def __init__(self):
        ##super(ColorObjectMatcher, self).__init__()
        super(TestDataSave, self).__init__()
        self.save_path = rospy.get_param('~save_path')
        self.dir_num = 0
        while os.path.exists(os.path.join(self.save_path,str(self.dir_num))):
            self.dir_num += 1
        
        
    def subscribe(self):
        self.sub_img = message_filters.Subscriber('~input', Image)
        self.sub_label = message_filters.Subscriber('~input/label', Image)
        queue_size = rospy.get_param('~queue_size', 100)
        if rospy.get_param('~approximate_sync', False):
            sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_img, self.sub_label], queue_size=queue_size,
                slop=0.1)
        else:
            sync = message_filters.TimeSynchronizer(
                [self.sub_img, self.sub_label], queue_size=queue_size)
        sync.registerCallback(self._save)


    def unsubscribe(self):
        self.sub_img.sub.unregister()
        self.sub_label.sub.unregister()


    def _save(self, img_msg, label_msg):

        print('press any key to save')
        raw_input()

        # make directory to save
        save_path = os.path.join(self.save_path + str(self.dir_num))
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path,'region/'))
        os.mkdir(os.path.join(save_path,'total/'))

        # convert image                                                                                    
        bridge = cv_bridge.CvBridge()
        input_image = bridge.imgmsg_to_cv2(img_msg, 'rgb8')
        input_label = bridge.imgmsg_to_cv2(label_msg)

        # regional image
        region_imgs = []
        for l in np.unique(input_label):
            if l == 0:  # bg_label 
                continue
            mask = (input_label == l)
            region = jsk_recognition_utils.bounding_rect_of_mask(
                input_image, mask)
            region_imgs.append(region)

        # save image
        cv2.imwrite(os.path.join(save_path,'total/')+'input.jpg',input_image)
        cv2.imwrite(os.path.join(save_path,'total/')+'label.jpg',input_label)
        i = 0
        for region in region_images :
            cv2.imwrite(os.path.join(save_path,'region/')+'{0}.jpg'.format(str(i)),region)
            i += 1

        self.dir_num += 1


if __name__ == "__main__":
    rospy.init_node('color_object_matcher')
    TestDataSave()
    rospy.spin()
