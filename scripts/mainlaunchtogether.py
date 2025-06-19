#!/usr/bin/env python3

import rospy
import cv2
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from face_recognizer import FaceRecognizer
from voice_interface import VoiceInterface
from conversation_manager import ConversationManager
from gemini_handler import GeminiHandler
from config_manager import ConfigManager

class IntegratedChatbot:
    def __init__(self):
        rospy.init_node('integrated_chatbot_node')
        
        # Load config and initialize components
        self.config = ConfigManager().get_config()
        self.face_recognizer = FaceRecognizer(self.config)
        self.voice_interface = VoiceInterface(self.config)
        self.gemini = GeminiHandler(self.config['google_api_key'])
        self.conversation = ConversationManager(
            professors_db=self.config['professors'],
            gemini_handler=self.gemini
        )
        
        # ROS & Camera setup
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/face_recognition/output_image", Image, queue_size=1)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera!")
            rospy.signal_shutdown("Camera unavailable")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State
        self.current_user = None
        self.last_recognition_time = 0
        self.conversation_active = False
        self.greeted = False
        self.last_greeted_user = None

    def _process_frame(self):
        """Process a single frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture frame")
            return

        frame = cv2.flip(frame, 1)
        recognition_results = self.face_recognizer.recognize_faces(frame)
        
        for result in recognition_results:
            self._draw_face_info(frame, result)
            self._handle_new_face_recognition(result)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

    def _draw_face_info(self, frame, result):
        """Draw rectangle and label on detected face"""
        name, emotion, gender, (x1, y1, x2, y2), confidence = result
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({gender}) - {emotion}"
        cv2.putText(frame, label, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _handle_new_face_recognition(self, result):
        """Set user and greet once when face is recognized"""
        name, _, _, _, _ = result
        current_time = time.time()
        min_interval = self.config.get('min_recognition_interval', 10)

        if name != "Unknown" and current_time - self.last_recognition_time > min_interval:
            if name != self.last_greeted_user:
                self.current_user = name
                self.last_recognition_time = current_time
                self.greeted = False
                self._greet_face()

    def _greet_face(self):
        """Speak greeting when a new face is recognized"""
        if self.current_user and not self.greeted:
            greeting = self.config['professors'].get(
                self.current_user,
                f"مرحباً {self.current_user}! من فضلك قل 'نور' للبدء."
            )
            self.voice_interface.speak(greeting)
            self.greeted = True
            self.last_greeted_user = self.current_user

    def _handle_conversation(self):
        """Handle the voice conversation flow"""
        user_input = self.voice_interface.listen()
        if not user_input:
            return

        # Wait for the wake word
        if not self.conversation_active:
            if "نور" in user_input:
                self.conversation_active = True
                self.greeted = False
                if not self.current_user:
                    self.current_user = self.config.get('default_name', 'زائر')
                rospy.loginfo("Conversation started with wake word.")
            return

        # Greet again at start of conversation (only once)
        if not self.greeted and self.current_user:
            greeting = self.config['professors'].get(
                self.current_user,
                f"مرحباً {self.current_user}! كيف يمكنني مساعدتك اليوم؟"
            )
            self.voice_interface.speak(greeting)
            self.greeted = True
            return

        # Get response and speak
        response = self.conversation.get_response(user_input)
        self.voice_interface.speak(response)

        # Handle end
        if self.conversation.should_end_conversation(user_input):
            self.voice_interface.speak("مع السلامة، أتمنى لك يومًا سعيدًا!")
            self.conversation_active = False
            self.current_user = None
            self.greeted = False
            self.last_greeted_user = None

    def _cleanup(self):
        """Cleanup on shutdown"""
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            try:
                self._process_frame()
                self._handle_conversation()
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
            rate.sleep()
        self._cleanup()

if __name__ == '__main__':
    try:
        IntegratedChatbot().run()
    except rospy.ROSInterruptException:
        pass

