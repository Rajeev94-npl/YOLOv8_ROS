#!/usr/bin/env python
import cv2
import torch
import random

import rospy

from cv_bridge import CvBridge

from ultralytics import YOLO


from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import SetBool


class Yolov8Node():

    def __init__(self) -> None:

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # params
        # self.model = rospy.get_param("model", "yolov8x.pt")
       
        # self.tracker = rospy.get_param("tracker", "bytetrack.yaml")

        # self.img_topic = rospy.get_param("img_topic", "/airsim_node/Drone_1/camera_1/Scene")
       

        # self.threshold = rospy.get_param("threshold", 0.5)

        # self.enable = rospy.get_param("enable", True)
        
        # # params
        self.model1 = "yolov8x.pt"
        self.model2 = "yolov8x-seg.pt"
        self.bool_yolo_seg = False

        self.img_topic = "/airsim_node/Drone_1/camera_1/Scene"
       
        self.threshold =  0.5

        self.enable =  True

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        if self.bool_yolo_seg:
            self.yolo = YOLO(self.model2)
            self.yolo.fuse()
        else:
            self.yolo = YOLO(self.model1)
            self.yolo.fuse()
        rospy.sleep(1)
        self.yolo.to(self.device)

        # topics
        self._pub = rospy.Publisher("detections",Detection2DArray, queue_size= 10)
        self._detectionimage_pub = rospy.Publisher("dbg_image", Image, queue_size= 10)
        rospy.sleep(1)
        self._sub = rospy.Subscriber(self.img_topic, Image, self.image_cb)


    def image_cb(self, msg: Image) -> None:

        if self.enable:

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            
            results = self.yolo.track(source=cv_image,show=False, tracker="bytetrack.yaml")
            

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()
            

            for b in results.boxes:

                label = self.yolo.names[int(b.cls)]
                score = float(b.conf)

                if score < self.threshold:
                    continue

                detection = Detection2D()

                detection.header = msg.header

                detection.source_img = msg

                box = b.xywh[0]

                # get boxes values
                detection.bbox.center.x = float(box[0])
                detection.bbox.center.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # get track id
                track_id = ""
                if not b.id is None:
                    track_id = int(b.id)
                #detection.id = track_id


                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(b.cls)
                hypothesis.score = score
                hypothesis.pose.pose.position.x = track_id
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b1 = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b1)
                color = self._class_to_color[label]

                min_pt = (round(detection.bbox.center.x - detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y - detection.bbox.size_y / 2.0))
                max_pt = (round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y + detection.bbox.size_y / 2.0))
                cv_image = cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{}:({}) {:.3f}".format(label, str(track_id), score)
                pos = (min_pt[0], max(15, int(min_pt[1] - 10)))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv_image = cv2.putText(cv_image, label, pos, font,
                            0.5, color, 1, cv2.LINE_AA)
                
                
                

                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self._pub.publish(detections_msg)
            
            
            if self.bool_yolo_seg:
                annotated_results = results[0].plot(conf=False, labels = False, img= cv_image,  boxes = False,masks = True)
                
            

                self._detectionimage_pub.publish(self.cv_bridge.cv2_to_imgmsg(annotated_results,
                                                                encoding=msg.encoding))
                cv2.imshow("Real-Time Detection with Tracking and Segmentation",annotated_results)
                cv2.waitKey(1)
            else:
                self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                                encoding=msg.encoding))
                cv2.imshow("Real-Time Detection with Tracking and Segmentation",cv_image)
                cv2.waitKey(1)

            #cv2.imshow("Real-Time Detection with Tracking",cv_image)
            #cv2.waitKey(1)

            if rospy.is_shutdown():
                cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("yolov8_sea", anonymous= True)
    Yolov8Node()
    rospy.spin()