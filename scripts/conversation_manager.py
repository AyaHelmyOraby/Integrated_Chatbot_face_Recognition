import rospy

class ConversationManager:
    def __init__(self, professors_db, gemini_handler):
        self.professors = {
            "مصطفى": "دكتور مصطفى الشافعي هو رئيس قسم هندسة الاتصالات وتكنولوجيا المعلومات بجامعة زويل. مكتبه في S5، منطقة D.",
            "mahmoud": "Dr Mahmoud Abdelaziz is a respected professor at Zewail University.",
            "محمود": "دكتور محمود عبد العزيز هو أستاذ في جامعة زويل.",
            "omar": "Dr Omar Fahmy is a professor at Zewail University.",
            "عمر": "دكتور عمر فهمي هو أستاذ في جامعة زويل."
        }
        self.gemini = gemini_handler
        self.current_user = None

    def get_response(self, user_input):
        """Generate appropriate response to user input"""
        user_input_lower = user_input.lower()
        
        # Check for "I'm fine" response
        if any(phrase in user_input_lower for phrase in ["بخير", "الحمد لله", "تمام", "انا بخير"]):
            return "الحمد لله، سعيد بسماع ذلك! كيف يمكنني مساعدتك؟"
        
        # Check for specific question about Dr. ElShafie
        if any(phrase in user_input_lower for phrase in ["مين هو دكتور الشافعي", "من هو دكتور الشافعي"]):
            return "دكتور مصطفى الشافعي هو رئيس قسم هندسة الاتصالات وتكنولوجيا المعلومات بجامعة زويل"
        
        # Check if this is a known professor
        for name, reply in self.professors.items():
            if name.lower() in user_input_lower:
                return reply
                
        # Use Gemini for general questions
        return self.gemini.get_response(user_input)

    def should_end_conversation(self, user_input):
        """Determine if conversation should end"""
        return any(word in user_input.lower() for word in ["شكرا", "مع السلامة", "وداعا"])
