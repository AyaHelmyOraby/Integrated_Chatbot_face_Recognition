#!/usr/bin/env python

import rospy
import cv2
from threading import Thread
from vision_module import VisionModule
from voice_module import VoiceModule
from utils import ConfigLoader, ROSPublisherManager
from std_msgs.msg import String

class VisionChatbot:
    def __init__(self):
        rospy.init_node('vision_chatbot_node')
        rospy.loginfo("\n🤖 Vision Chatbot - ROS Node Initialized")

        self.config = ConfigLoader().load()
        self.publishers = ROSPublisherManager()
        self.vision = VisionModule(self.config, self.publishers)
        self.voice = VoiceModule(self.config, self.publishers)

        rospy.Subscriber("/vision_chatbot/speech/input", String, self.voice.speech_input_callback)
        rospy.Subscriber("/vision_chatbot/user_name", String, self.vision.name_callback)

        rospy.on_shutdown(self.cleanup)
        Thread(target=self.voice.listen_loop, daemon=True).start()

    def cleanup(self):
        self.vision.release()
        cv2.destroyAllWindows()
        rospy.loginfo("\n🚭 Shutting down Vision Chatbot")

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            frame = self.vision.get_frame()
            if frame is not None:
                results = self.vision.process_frame(frame)
                self.vision.publish_data(frame, results)
            rate.sleep()

if __name__ == '__main__':
    try:
        bot = VisionChatbot()
        bot.run()
    except rospy.ROSInterruptException:
        pass

