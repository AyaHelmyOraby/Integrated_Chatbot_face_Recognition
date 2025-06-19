import google.generativeai as genai
import rospy

class GeminiHandler:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_response(self, user_input):
        """Get response from Gemini API"""
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
            response = self.model.generate_content(arabic_prompt)
            arabic_response = response.text.strip()
            arabic_response = arabic_response.replace("*", "").replace("**", "")
            if any(c in arabic_response for c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
                return arabic_response
            else:
                return "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك إعادة صياغته؟"
        except Exception as e:
            rospy.logerr(f"API Error: {str(e)}")
            return "حدث خطأ تقني، يرجى المحاولة مرة أخرى لاحقًا"
