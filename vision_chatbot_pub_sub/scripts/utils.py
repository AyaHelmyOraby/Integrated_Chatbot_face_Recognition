# utils.py

import os
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

class ConfigLoader:
    def load(self):
        pkg_path = os.path.dirname(os.path.abspath(__file__))
        return {
            'image_folder': rospy.get_param('~image_folder', os.path.join(pkg_path, 'data')),
            'emotion_model_path': rospy.get_param('~emotion_model', os.path.join(pkg_path, 'models', 'emotion-ferplus-8.onnx')),
            'gender_proto': rospy.get_param('~gender_proto', os.path.join(pkg_path, 'models', 'gender_deploy.prototxt')),
            'gender_model': rospy.get_param('~gender_model', os.path.join(pkg_path, 'models', 'gender_net.caffemodel')),
            'speech_timeout': rospy.get_param('~speech_timeout', 5),
            'min_recognition_interval': rospy.get_param('~min_recognition_interval', 10),
            'professors': {
                "Aya Oraby": "مرحباً دكتورة آية! كيف يمكنني مساعدتك اليوم؟",
                "Yasmeen Abosaif": "أهلاً دكتورة ياسمين! كيف حالك؟",
                "Prof. Elshafei": "تحية طيبة بروفيسور الشافعي! كيف يمكنني مساعدتك؟",
                "Dr. Samy Soliman": "أهلاً وسهلاً دكتور سامي! كيف حالك اليوم؟"
            }
        }

class ROSPublisherManager:
    def __init__(self):
        self.face_image = rospy.Publisher("/vision_chatbot/face_image", Image, queue_size=1)
        self.face_name = rospy.Publisher("/vision_chatbot/face/name", String, queue_size=1)
        self.face_gender = rospy.Publisher("/vision_chatbot/face/gender", String, queue_size=1)
        self.face_emotion = rospy.Publisher("/vision_chatbot/face/emotion", String, queue_size=1)
        self.face_confidence = rospy.Publisher("/vision_chatbot/face/confidence", String, queue_size=1)
        self.speech = rospy.Publisher("/vision_chatbot/speech/output", String, queue_size=1)

