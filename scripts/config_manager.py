import os
import rospy
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self):
        self.pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv()
        
    def get_config(self):
        return {
                'robot_name': rospy.get_param('~robot_name', 'نور'),  
            'image_folder': rospy.get_param('~image_folder', os.path.join(self.pkg_path, 'data')),
            'emotion_model': rospy.get_param('~emotion_model', os.path.join(self.pkg_path, 'models', 'emotion-ferplus-8.onnx')),
            'gender_proto': rospy.get_param('~gender_proto', os.path.join(self.pkg_path, 'models', 'gender_deploy.prototxt')),
            'gender_model': rospy.get_param('~gender_model', os.path.join(self.pkg_path, 'models', 'gender_net.caffemodel')),
            'speech_timeout': rospy.get_param('~speech_timeout', 5),
            'min_recognition_interval': rospy.get_param('~min_recognition_interval', 10),
            'google_api_key': os.getenv("GOOGLE_API_KEY"),
            'professors': {
                "مصطفى": "مرحباً دكتور مصطفى! كيف يمكنني مساعدتك اليوم؟",
                "الشافعي": "مرحباً دكتور الشافعي! كيف حالك اليوم؟",
                "محمود": "أهلاً وسهلاً دكتور محمود! كيف يمكنني خدمتك؟",
                "عمر": "تحية طيبة دكتور عمر! ما الذي تحتاج إليه؟",
                "Aya Oraby": "مرحباً  آية! كيف يمكنني مساعدتك اليوم؟",
                "Yasmeen Abosaif": "أهلاً دكتورة ياسمين! كيف حالك؟",
                "Prof. Elshafei": "تحية طيبة بروفيسور الشافعي! كيف يمكنني مساعدتك؟",
                "Dr. Mahmoud Abdelaziz": "مرحباً دكتور محمود! كيف يمكنني مساعدتك؟",
                "Dr. Samy Soliman": "مرحباً دكتور سامي! كيف حالك اليوم؟"
            }
        }
