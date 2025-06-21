#!/usr/bin/env python

import rospy
import cv2
import os
import numpy as np
import onnxruntime as ort
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
import tempfile
import time
from threading import Lock, Thread
import google.generativeai as genai
from dotenv import load_dotenv

class VisionChatbot:
    def __init__(self):
        rospy.init_node('vision_chatbot_node')

        rospy.loginfo("""
==================================
🤖 Vision Chatbot - ROS Node v1.0
🗕 Startup Time: %s
==================================
""" % time.strftime("%Y-%m-%d %H:%M:%S"))

        self.window_name = "Vision Chatbot"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        self.pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.load_config()
        self.init_face_recognition()
        self.init_voice_components()

        self.bridge = CvBridge()
        self.current_user = None
        self.last_recognition_time = 0
        self.conversation_active = False
        self.state_lock = Lock()
        self.override_name = None

        self.face_image_pub = rospy.Publisher("/vision_chatbot/face_image", Image, queue_size=1)
        self.face_name_pub = rospy.Publisher("/vision_chatbot/face/name", String, queue_size=1)
        self.face_gender_pub = rospy.Publisher("/vision_chatbot/face/gender", String, queue_size=1)
        self.face_emotion_pub = rospy.Publisher("/vision_chatbot/face/emotion", String, queue_size=1)
        self.face_confidence_pub = rospy.Publisher("/vision_chatbot/face/confidence", String, queue_size=1)
        self.speech_pub = rospy.Publisher("/vision_chatbot/speech/output", String, queue_size=1)
        rospy.Subscriber("/vision_chatbot/speech/input", String, self.speech_input_callback)
        rospy.Subscriber("/vision_chatbot/user_name", String, self.name_callback)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("❌ Cannot open camera")
            rospy.signal_shutdown("Camera unavailable")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        rospy.loginfo("📷 Camera initialized (640x480)")
        rospy.loginfo(f"📏 Camera resolution: {self.cap.get(3)}x{self.cap.get(4)}")

        rospy.loginfo("✅ Vision Chatbot Node Initialized")
        rospy.on_shutdown(self.cleanup)

        Thread(target=self.listen_for_user_input, daemon=True).start()
        self.main_loop()

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("🚭 Shutting down vision chatbot node")

    def load_config(self):
        self.image_folder = rospy.get_param('~image_folder', os.path.join(self.pkg_path, 'data'))
        self.emotion_model_path = rospy.get_param('~emotion_model', os.path.join(self.pkg_path, 'models', 'emotion-ferplus-8.onnx'))
        self.gender_proto = rospy.get_param('~gender_proto', os.path.join(self.pkg_path, 'models', 'gender_deploy.prototxt'))
        self.gender_model = rospy.get_param('~gender_model', os.path.join(self.pkg_path, 'models', 'gender_net.caffemodel'))
        self.speech_timeout = rospy.get_param('~speech_timeout', 5)
        self.min_recognition_interval = rospy.get_param('~min_recognition_interval', 10)

        self.professors = {
            "Aya Oraby": "مرحباً دكتورة آية! كيف يمكنني مساعدتك اليوم؟",
            "Yasmeen Abosaif": "أهلاً دكتورة ياسمين! كيف حالك؟",
            "Prof. Elshafei": "تحية طيبة بروفيسور الشافعي! كيف يمكنني مساعدتك؟",
            "Dr. Samy Soliman": "أهلاً وسهلاً دكتور سامي! كيف حالك اليوم؟"
        }

    def init_face_recognition(self):
        self.emotion_labels = ['surprise', 'neutral', 'happy', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        self.gender_labels = ['Male', 'Female']

        self.emotion_session = ort.InferenceSession(self.emotion_model_path)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.known_face_names = []
        face_samples, ids = [], []
        id_count = 0

        rospy.loginfo("🧠 Loading known faces...")
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
                    name = os.path.splitext(file)[0]
                    self.known_face_names.append(name)
                    rospy.loginfo(f"👤 Loaded face: {name}")
                    id_count += 1

        if face_samples:
            self.face_recognizer.train(face_samples, np.array(ids))
            rospy.loginfo(f"📚 Trained with {len(face_samples)} face(s)")
        else:
            rospy.logwarn("⚠️ No valid face data found!")

    def init_voice_components(self):
        self.speech_recognizer = sr.Recognizer()
        os.environ['ALSA_DEBUG'] = '0'
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    def main_loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("⚠️ Failed to read from camera")
                continue
            frame = cv2.flip(frame, 1)
            results = self.recognize_faces(frame)
            self.publish_face_data(frame, results)
            rate.sleep()

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                id, conf = self.face_recognizer.predict(cv2.resize(roi, (200, 200)))
                name = self.known_face_names[id] if conf < 80 else "Unknown"
            except:
                name, conf = "Unknown", 100
            if self.override_name and name == "Unknown":
                name = self.override_name
            emotion = self.detect_emotion(roi)
            gender = self.detect_gender(frame[y:y+h, x:x+w])
            results.append((name, emotion, gender, (x, y, x+w, y+h), conf))
        return results

    def publish_face_data(self, frame, results):
        annotated = frame.copy()
        for name, emotion, gender, (x1, y1, x2, y2), conf in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = name if name != "Unknown" else (self.override_name if self.override_name else "Unknown")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} ({gender}) - {emotion}", (x1+5, y2-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.face_name_pub.publish(String(label))
            self.face_gender_pub.publish(String(gender))
            self.face_emotion_pub.publish(String(emotion))
            self.face_confidence_pub.publish(String(str(conf)))
            self.check_conversation_start(label)
        cv2.imshow(self.window_name, annotated)
        cv2.waitKey(1)
        self.face_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))

    def detect_emotion(self, face_img):
        face_img = cv2.resize(face_img, (64, 64)).astype(np.float32)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=0)
        ort_inputs = {self.emotion_session.get_inputs()[0].name: face_img}
        ort_outs = self.emotion_session.run(None, ort_inputs)
        logits = ort_outs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return self.emotion_labels[np.argmax(probs)]

    def detect_gender(self, face_img):
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(face_img, (227, 227)), 1.0,
                                     (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.gender_net.setInput(blob)
        return self.gender_labels[np.argmax(self.gender_net.forward())]

    def listen_for_user_input(self):
        while not rospy.is_shutdown():
            with self.state_lock:
                if not self.conversation_active:
                    continue
            try:
                with sr.Microphone() as source:
                    rospy.loginfo("🎤 Listening for user input...")
                    self.speech_recognizer.adjust_for_ambient_noise(source)
                    audio = self.speech_recognizer.listen(source, timeout=self.speech_timeout)
                    user_input = self.speech_recognizer.recognize_google(audio, language="ar-EG")
                    if user_input:
                        rospy.loginfo(f"🗣️ You said: {user_input}")
                        self.speech_input_callback(String(user_input))
            except Exception as e:
                rospy.logwarn(f"🎤 Listening error: {e}")

    def speech_input_callback(self, msg):
        user_input = msg.data
        rospy.loginfo(f"🗣️ User said: {user_input}")
        if self.current_user in self.professors:
            greeting = self.professors[self.current_user]
        else:
            greeting = f"مرحباً {self.current_user}! كيف يمكنني مساعدتك؟"
        prompt = f"{greeting}\n\n{user_input}"
        try:
            response = self.gemini_model.generate_content(prompt)
            reply_text = response.text
        except Exception as e:
            rospy.logerr(f"❌ Error generating response: {e}")
            reply_text = "حدث خطأ أثناء المعالجة. حاول مرة أخرى لاحقاً."
        rospy.loginfo(f"🤖 Reply: {reply_text}")
        self.speak(reply_text)
        self.speech_pub.publish(String(reply_text))
        self.conversation_active = False

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='ar')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                audio = AudioSegment.from_file(fp.name, format="mp3")
                play(audio)
                os.unlink(fp.name)
        except Exception as e:
            rospy.logerr(f"🔊 Failed to speak: {e}")

    def name_callback(self, msg):
        self.override_name = msg.data

    def check_conversation_start(self, name):
        current_time = time.time()
        if name != "Unknown" and not self.conversation_active and \
           current_time - self.last_recognition_time > self.min_recognition_interval:
            self.current_user = name
            self.last_recognition_time = current_time
            self.conversation_active = True
            greeting = self.professors.get(name, f"مرحباً {name}!")
            self.speak(greeting)

if __name__ == '__main__':
    try:
        VisionChatbot()
    except rospy.ROSInterruptException:
        pass

