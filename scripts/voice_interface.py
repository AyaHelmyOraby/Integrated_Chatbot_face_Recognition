import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import arabic_reshaper
from bidi.algorithm import get_display
import os
import rospy

class VoiceInterface:
    def __init__(self, config):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        os.environ['ALSA_DEBUG'] = '0'

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
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.config['speech_timeout'], 
                    phrase_time_limit=10
                )
                user_input = self.recognizer.recognize_google(audio, language="ar-EG")
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
