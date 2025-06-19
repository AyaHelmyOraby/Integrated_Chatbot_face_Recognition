#!/home/philomath/catkin_ws/venvs/noor_venv_py39_clean/bin/python

import rospy
import cv2
import os
import numpy as np
import onnxruntime as ort
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import arabic_reshaper
from bidi.algorithm import get_display
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tempfile
import time
from threading import Lock
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

class IntegratedChatbot:
    def __init__(self):
        rospy.init_node('integrated_chatbot_node')
        
        # Load configuration
        self.pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.load_config()
        
        # Initialize face recognition components
        self.init_face_recognition()
        
        # Initialize voice chatbot components
        self.init_voice_chatbot()
        
        # State management
        self.current_user = None
        self.last_recognition_time = 0
        self.conversation_active = False
        self.state_lock = Lock()
        self.greeted = False
        
        # ROS publishers/subscribers
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/face_recognition/output_image", Image, queue_size=1)
        self.face_name_pub = rospy.Publisher("/detected_face_name", String, queue_size=1)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera!")
            rospy.signal_shutdown("Camera unavailable")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        rospy.loginfo("Integrated Chatbot Node Initialized")

    def load_config(self):
        """Load configuration parameters"""
        self.image_folder = rospy.get_param('~image_folder', os.path.join(self.pkg_path, 'data'))
        self.emotion_model_path = rospy.get_param('~emotion_model', os.path.join(self.pkg_path, 'models', 'emotion-ferplus-8.onnx'))
        self.gender_proto = rospy.get_param('~gender_proto', os.path.join(self.pkg_path, 'models', 'gender_deploy.prototxt'))
        self.gender_model = rospy.get_param('~gender_model', os.path.join(self.pkg_path, 'models', 'gender_net.caffemodel'))
        
        # Voice parameters
        self.speech_timeout = rospy.get_param('~speech_timeout', 5)
        self.min_recognition_interval = rospy.get_param('~min_recognition_interval', 10)
        
        # Professors database
        self.professors = {
            "مصطفى": "مرحباً دكتور مصطفى! كيف يمكنني مساعدتك اليوم؟",
            "الشافعي": "مرحباً دكتور الشافعي! كيف حالك اليوم؟",
            "محمود": "أهلاً وسهلاً دكتور محمود! كيف يمكنني خدمتك؟",
            "عمر": "تحية طيبة دكتور عمر! ما الذي تحتاج إليه؟",
            "Aya Oraby": "مرحباً دكتورة آية! كيف يمكنني مساعدتك اليوم؟",
            "Yasmeen Abosaif": "أهلاً دكتورة ياسمين! كيف حالك؟",
            "Prof. Elshafei": "تحية طيبة بروفيسور الشافعي! كيف يمكنني مساعدتك؟",
            "Dr. Mahmoud Abdelaziz": "مرحباً دكتور محمود! كيف يمكنني خدمتك؟",
            "Dr. Samy Soliman": "أهلاً وسهلاً دكتور سامي! كيف حالك اليوم؟"
        }

    def init_face_recognition(self):
        """Initialize face recognition models and components"""
        # Emotion and Gender labels
        self.emotion_labels = ['surprise', 'neutral', 'happy', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        self.gender_labels = ['Male', 'Female']
        
        # Load models with error handling
        try:
            if not os.path.exists(self.emotion_model_path):
                raise FileNotFoundError(f"Emotion model not found at {self.emotion_model_path}")
            if not os.path.exists(self.gender_model):
                raise FileNotFoundError(f"Gender model not found at {self.gender_model}")
            if not os.path.exists(self.gender_proto):
                raise FileNotFoundError(f"Gender proto not found at {self.gender_proto}")
                
            self.emotion_session = ort.InferenceSession(self.emotion_model_path)
            self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
            rospy.loginfo("Models loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load models: {str(e)}")
            raise
            
        # Face recognition setup
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_face_names = []
        self.load_known_faces()

    def init_voice_chatbot(self):
        """Initialize voice recognition and synthesis components"""
        self.speech_recognizer = sr.Recognizer()
        self.speech_recognizer.pause_threshold = 0.8
        os.environ['ALSA_DEBUG'] = '0'

    def load_known_faces(self):
        """Load known faces from the database"""
        face_samples = []
        ids = []
        id_count = 0

        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        continue

                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) == 0:
                        continue

                    (x, y, w, h) = faces[0]
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    face_samples.append(face_roi)
                    ids.append(id_count)
                    self.known_face_names.append(os.path.splitext(file)[0])
                    id_count += 1
                    rospy.loginfo(f"Loaded face: {self.known_face_names[-1]}")

        if face_samples:
            self.face_recognizer.train(face_samples, np.array(ids))
            rospy.loginfo(f"Face recognizer trained with {len(face_samples)} faces")
        else:
            rospy.logwarn("No faces found for training!")

    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            results = []

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Face recognition
                try:
                    id, confidence = self.face_recognizer.predict(cv2.resize(face_roi, (200, 200)))
                    name = self.known_face_names[id] if confidence < 70 else "Unknown"
                except Exception as e:
                    rospy.logwarn(f"Face prediction failed: {str(e)}")
                    name = "Unknown"
                    confidence = 100
                
                # Emotion detection
                emotion = self.detect_emotion(face_roi)
                
                # Gender detection
                gender = self.detect_gender(frame[y:y+h, x:x+w])
                
                results.append((name, emotion, gender, (x, y, x+w, y+h), confidence))
            
            return results
        except Exception as e:
            rospy.logerr(f"Face recognition failed: {str(e)}")
            return []

    def detect_emotion(self, face_img):
        """Detect emotion from face image"""
        try:
            face_img = cv2.resize(face_img, (64, 64)).astype(np.float32)
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=0)

            ort_inputs = {self.emotion_session.get_inputs()[0].name: face_img}
            ort_outs = self.emotion_session.run(None, ort_inputs)
            logits = ort_outs[0][0]

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            return self.emotion_labels[np.argmax(probs)]
        except Exception as e:
            rospy.logwarn(f"Emotion detection failed: {str(e)}")
            return "N/A"

    def detect_gender(self, face_img):
        """Detect gender from face image"""
        try:
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                
            blob = cv2.dnn.blobFromImage(
                cv2.resize(face_img, (227, 227)),
                scalefactor=1.0,
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            self.gender_net.setInput(blob)
            return self.gender_labels[np.argmax(self.gender_net.forward())]
        except Exception as e:
            rospy.logwarn(f"Gender detection failed: {str(e)}")
            return "N/A"

    def speak(self, text):
        """Convert text to Arabic speech"""
        try:
            tts = gTTS(text, lang="ar")
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio = AudioSegment.from_file(fp.name, format="mp3")
                play(audio)
        except Exception as e:
            rospy.logerr(f"Error in speech generation: {str(e)}")

    def listen(self):
        """Listen for Arabic voice input"""
        with sr.Microphone() as source:
            rospy.loginfo("🎤 Ready to listen... (Please speak now)")
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.speech_recognizer.listen(source, timeout=self.speech_timeout, 
                                                     phrase_time_limit=10)
                user_input = self.speech_recognizer.recognize_google(audio, language="ar-EG")
                reshaped = arabic_reshaper.reshape(user_input)
                bidi_text = get_display(reshaped)
                rospy.loginfo(f"🗣️ Recognized: {bidi_text}")
                return user_input
            except sr.WaitTimeoutError:
                rospy.loginfo("⏳ Listening timeout")
                return None
            except sr.UnknownValueError:
                rospy.loginfo("❌ Speech not recognized")
                self.speak("لم أتمكن من فهم ما تقول. يرجى التحدث بوضوح.")
                return None
            except sr.RequestError as e:
                rospy.logerr(f"❌ Speech recognition error: {e}")
                self.speak("حدث خطأ في خدمة التعرف على الصوت. يرجى المحاولة لاحقًا.")
                return None

    def get_gemini_response(self, user_input):
        """Get response from Gemini API for general questions"""
        arabic_prompt = f"""
        أنت مساعد عربي يتحدث باللغة العربية الفصحى فقط.
        أجب على السؤال التالي بلغة عربية واضحة وسليمة:
        {user_input}
        
        يجب أن تكون الإجابة:
        - باللغة العربية فقط
        - واضحة ومباشرة
        - بدون أحرف خاصة مثل * أو #
        - إذا كان السؤال غير واضح، قل: "لم أفهم سؤالك، هل يمكنك إعادة صياغته؟"
        """

        try:
            response = GEMINI_MODEL.generate_content(arabic_prompt)
            arabic_response = response.text.strip()
            arabic_response = arabic_response.replace("*", "").replace("**", "")
            if any(c in arabic_response for c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
                return arabic_response
            else:
                return "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك إعادة صياغته؟"
        except Exception as e:
            rospy.logerr(f"API Error: {str(e)}")
            return "حدث خطأ تقني، يرجى المحاولة مرة أخرى لاحقًا"

    def get_response(self, user_input):
        """Generate appropriate response to user input"""
        user_input_lower = user_input.lower()
        
        # Check for "I'm fine" response
        if any(phrase in user_input_lower for phrase in ["بخير", "الحمد لله", "تمام", "انا بخير"]):
            return "الحمد لله، سعيد بسماع ذلك! كيف يمكنني مساعدتك؟"
        
        # Check if this is a known professor
        for name, reply in self.professors.items():
            if name.lower() in user_input_lower:
                return reply
                
        # Use Gemini for general questions
        return self.get_gemini_response(user_input)

    def handle_face_recognition(self):
        """Process camera frames for face recognition"""
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture frame")
            return

        frame = cv2.flip(frame, 1)
        recognition_results = self.recognize_faces(frame)
        
        with self.state_lock:
            for name, emotion, gender, (x1, y1, x2, y2), confidence in recognition_results:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"{name} ({gender}) - {emotion}"
                cv2.putText(frame, text, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # If we recognize someone and not in conversation
                current_time = time.time()
                if (name != "Unknown" and 
                    not self.conversation_active and 
                    current_time - self.last_recognition_time > self.min_recognition_interval):
                    
                    self.current_user = name
                    self.last_recognition_time = current_time
                    self.conversation_active = True
                    self.greeted = False

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

    def handle_conversation(self):
        """Handle voice conversation when active"""
        with self.state_lock:
            if not self.conversation_active:
                return
                
        # First greet the user if we haven't already
        if not self.greeted and self.current_user:
            greeting = self.professors.get(self.current_user, f"مرحباً {self.current_user}! كيف حالك اليوم؟")
            self.speak(greeting)
            with self.state_lock:
                self.greeted = True
            return
            
        # Then listen for their response
        user_input = self.listen()
        if user_input:
            response = self.get_response(user_input)
            self.speak(response)
            
            # End conversation if appropriate
            if any(word in user_input.lower() for word in ["شكرا", "مع السلامة", "وداعا"]):
                with self.state_lock:
                    self.conversation_active = False
                    self.current_user = None
                    self.greeted = False
                self.speak("مع السلامة، أتمنى لك يومًا سعيدًا!")

    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            try:
                self.handle_face_recognition()
                self.handle_conversation()
            except Exception as e:
                rospy.logerr(f"Error in main loop: {str(e)}")
            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        IntegratedChatbot().run()
    except rospy.ROSInterruptException:
        pass
