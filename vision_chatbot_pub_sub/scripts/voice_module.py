# voice_module.py

import rospy
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai

class VoiceModule:
    def __init__(self, config, publishers):
        self.config = config
        self.publishers = publishers
        self.recognizer = sr.Recognizer()
        os.environ['ALSA_DEBUG'] = '0'

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def listen_loop(self):
        while not rospy.is_shutdown():
            try:
                with sr.Microphone() as source:
                    rospy.loginfo("🎤 Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, timeout=self.config['speech_timeout'])
                    text = self.recognizer.recognize_google(audio, language="ar-EG")
                    if text:
                        self.speech_input_callback(rospy.msg.String(text))
            except Exception as e:
                rospy.logwarn(f"🎤 Error during listen: {e}")

    def speech_input_callback(self, msg):
        user_input = msg.data
        rospy.loginfo(f"🗣️ Input: {user_input}")

        from main import VisionChatbot  # late import to avoid circular import
        current_user = VisionChatbot().vision.current_user or "صديقي"
        greeting = self.config['professors'].get(current_user, f"مرحباً {current_user}! كيف يمكنني مساعدتك؟")
        prompt = f"{greeting}\n\n{user_input}"

        try:
            response = self.model.generate_content(prompt)
            reply = response.text
        except Exception as e:
            rospy.logerr(f"❌ Gemini error: {e}")
            reply = "حدث خطأ أثناء المعالجة. حاول لاحقاً."

        rospy.loginfo(f"🤖 Response: {reply}")
        speak(reply)
        self.publishers.speech.publish(reply)


def speak(text):
    try:
        tts = gTTS(text=text, lang='ar')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            audio = AudioSegment.from_file(fp.name, format="mp3")
            play(audio)
            os.unlink(fp.name)
    except Exception as e:
        rospy.logerr(f"🔊 Failed to speak: {e}")

