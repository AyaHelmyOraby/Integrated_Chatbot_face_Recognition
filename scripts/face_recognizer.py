import cv2
import numpy as np
import onnxruntime as ort
import rospy
import os

class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.emotion_labels = ['surprise', 'neutral', 'happy', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        self.gender_labels = ['Male', 'Female']
        self.known_face_names = []
        self._load_models()
        self._init_face_recognition()
        self._load_known_faces()

    def _load_models(self):
        """Load emotion and gender detection models"""
        self.emotion_session = ort.InferenceSession(self.config['emotion_model'])
        self.gender_net = cv2.dnn.readNet(self.config['gender_model'], self.config['gender_proto'])

    def _init_face_recognition(self):
        """Initialize face recognition components"""
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _load_known_faces(self):
        """Load known faces from the database"""
        face_samples = []
        ids = []
        id_count = 0

        if not os.path.exists(self.config['image_folder']):
            rospy.logwarn(f"Image folder not found: {self.config['image_folder']}")
            return

        for root, _, files in os.walk(self.config['image_folder']):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self._process_face_image(root, file, face_samples, ids, id_count)
                    id_count += 1

        if face_samples:
            self.face_recognizer.train(face_samples, np.array(ids))
            rospy.loginfo(f"Face recognizer trained with {len(face_samples)} faces")

    def _process_face_image(self, root, file, face_samples, ids, id_count):
        """Process a single face image for training"""
        img_path = os.path.join(root, file)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return

        (x, y, w, h) = faces[0]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        face_samples.append(face_roi)
        ids.append(id_count)
        self.known_face_names.append(os.path.splitext(file)[0])
        rospy.loginfo(f"Loaded face: {self.known_face_names[-1]}")

    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            results = []

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Face recognition
                name, confidence = self._recognize_face(face_roi)
                
                # Emotion detection
                emotion = self._detect_emotion(face_roi)
                
                # Gender detection
                gender = self._detect_gender(frame[y:y+h, x:x+w])
                
                results.append((name, emotion, gender, (x, y, x+w, y+h), confidence))
            
            return results
        except Exception as e:
            rospy.logerr(f"Face recognition failed: {str(e)}")
            return []

    def _recognize_face(self, face_roi):
        """Recognize a single face"""
        try:
            id, confidence = self.face_recognizer.predict(cv2.resize(face_roi, (200, 200)))
            return (self.known_face_names[id] if confidence < 70 else "Unknown", confidence)
        except Exception as e:
            rospy.logwarn(f"Face prediction failed: {str(e)}")
            return ("Unknown", 100)

    def _detect_emotion(self, face_img):
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

    def _detect_gender(self, face_img):
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
