#!/usr/bin/env python3

import rospy
import cv2
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from face_recognizer import FaceRecognizer
from config_manager import ConfigManager

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node')
        
        self.config = ConfigManager().get_config()
        self.face_recognizer = FaceRecognizer(self.config)
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = rospy.Publisher("/face_recognition/output_image", Image, queue_size=1)
        self.face_pub = rospy.Publisher("/recognized_face", String, queue_size=1)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera!")
            rospy.signal_shutdown("Camera unavailable")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.last_recognition_time = 0

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture frame")
            return None

        frame = cv2.flip(frame, 1)
        recognition_results = self.face_recognizer.recognize_faces(frame)
        
        for (name, emotion, gender, bbox, confidence) in recognition_results:
            self.draw_face_info(frame, name, emotion, gender, bbox)
            if name != "Unknown":
                current_time = time.time()
                if current_time - self.last_recognition_time > self.config['min_recognition_interval']:
                    self.face_pub.publish(String(name))
                    self.last_recognition_time = current_time
        
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        return frame

    def draw_face_info(self, frame, name, emotion, gender, bbox):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name} ({gender}) - {emotion}"
        cv2.putText(frame, text, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                frame = self.process_frame()
                if frame is not None:
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                rospy.logerr(f"Error: {str(e)}")
            rate.sleep()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
