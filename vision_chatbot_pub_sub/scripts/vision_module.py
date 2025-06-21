# vision_module.py

import cv2
import os
import numpy as np
import onnxruntime as ort
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class VisionModule:
    def __init__(self, config, publishers):
        self.config = config
        self.publishers = publishers
        self.bridge = CvBridge()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.gender_net = cv2.dnn.readNet(config['gender_model'], config['gender_proto'])
        self.emotion_session = ort.InferenceSession(config['emotion_model_path'])

        self.emotion_labels = ['surprise', 'neutral', 'happy', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        self.gender_labels = ['Male', 'Female']

        self.known_face_names = []
        self.override_name = None
        self.last_recognition_time = 0
        self.conversation_active = False
        self.current_user = None

        self.professors = config['professors']

        self._load_faces()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _load_faces(self):
        samples, ids = [], []
        idx = 0
        for file in os.listdir(self.config['image_folder']):
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(self.config['image_folder'], file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
                if len(faces):
                    (x, y, w, h) = faces[0]
                    roi = cv2.resize(img[y:y+h, x:x+w], (200, 200))
                    samples.append(roi)
                    ids.append(idx)
                    self.known_face_names.append(os.path.splitext(file)[0])
                    idx += 1
        if samples:
            self.face_recognizer.train(samples, np.array(ids))

    def get_frame(self):
        ret, frame = self.cap.read()
        return cv2.flip(frame, 1) if ret else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            try:
                id_, conf = self.face_recognizer.predict(roi)
                name = self.known_face_names[id_] if conf < 80 else "Unknown"
            except:
                name, conf = "Unknown", 100
            if self.override_name and name == "Unknown":
                name = self.override_name
            emotion = self._detect_emotion(roi)
            gender = self._detect_gender(frame[y:y+h, x:x+w])
            results.append((name, emotion, gender, (x, y, x+w, y+h), conf))
        return results

    def _detect_emotion(self, face_img):
        img = cv2.resize(face_img, (64, 64)).astype(np.float32)
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
        logits = self.emotion_session.run(None, {self.emotion_session.get_inputs()[0].name: img})[0][0]
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return self.emotion_labels[np.argmax(probs)]

    def _detect_gender(self, face_img):
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(face_img, (227, 227)), 1.0,
                                     (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
        self.gender_net.setInput(blob)
        return self.gender_labels[np.argmax(self.gender_net.forward())]

    def publish_data(self, frame, results):
        for name, emotion, gender, (x1, y1, x2, y2), conf in results:
            self.publishers.face_name.publish(name)
            self.publishers.face_gender.publish(gender)
            self.publishers.face_emotion.publish(emotion)
            self.publishers.face_confidence.publish(str(conf))
            self._start_convo(name)
            label = f"{name} ({gender}) - {emotion}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        self.publishers.face_image.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imshow("Vision Chatbot", frame)
        cv2.waitKey(1)

    def name_callback(self, msg):
        self.override_name = msg.data

    def _start_convo(self, name):
        now = time.time()
        if name != "Unknown" and not self.conversation_active and \
           now - self.last_recognition_time > self.config['min_recognition_interval']:
            self.current_user = name
            self.last_recognition_time = now
            self.conversation_active = True
            greeting = self.professors.get(name, f"مرحباً {name}!")
            from voice_module import speak
            speak(greeting)

