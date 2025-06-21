# 🤖 Integrated Chatbot with Face Recognition and Voice Assistant (Arabic)

This project is an **AI-powered voice chatbot integrated with facial recognition and emotion/gender analysis**, built using **ROS (Robot Operating System)**. It supports **Arabic speech recognition and synthesis**, personalized greetings based on known faces, and intelligent responses using **Google Gemini AI (gemini-2.0-flash)**.

---

## 🧠 Features

- 🎭 Real-time **face recognition** using OpenCV and LBPH algorithm
- 🧠 **Emotion detection** with ONNX emotion model
- 🧑‍🤝‍🧑 **Gender classification** using Caffe model
- 🎙️ **Voice recognition** with SpeechRecognition and Google STT API (Arabic)
- 🔊 **Text-to-speech (TTS)** in Arabic using gTTS + pydub
- 💬 Arabic **natural language interaction** using Google Gemini (Gemini 2.0 Flash)
- 🧾 Personalized greetings for professors and known faces
- 🔐 Thread-safe conversational state management using Python's `threading.Lock`

---

## 🛠️ Requirements

- Python 3.9
- ROS Noetic (tested)
- OpenCV with contrib modules (`cv2.face`)
- `gTTS`, `pydub`, `speechrecognition`, `arabic_reshaper`, `bidi`, `onnxruntime`, `google-generativeai`
- Microphone and camera access
- `.env` file with your Google Gemini API key

---

## 📁 Folder Structure

```
catkin_ws/
└── src/
    └── integrated_chatbot/
        ├── scripts/
        │   └── integrated_chatbot_node.py
        ├── models/
        │   ├── emotion-ferplus-8.onnx
        │   ├── gender_net.caffemodel
        │   └── gender_deploy.prototxt
        └── data/
            └── known_faces/
```

---

## ⚙️ Setup Instructions

1. **Clone and build the workspace**:
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

2. **Create and activate Python virtual environment**:
    ```bash
    python3.9 -m venv ~/catkin_ws/venvs/noor_venv_py39_clean
    source ~/catkin_ws/venvs/noor_venv_py39_clean/bin/activate
    ```

3. **Install required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

    _Sample `requirements.txt`:_
    ```
    opencv-contrib-python
    onnxruntime
    SpeechRecognition
    gTTS
    pydub
    arabic_reshaper
    python-bidi
    google-generativeai
    python-dotenv
    ```

4. **Create `.env` file in `scripts/` directory**:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

---

## 🚀 Running the Chatbot

1. **Launch your chatbot node**:
    ```bash
    rosrun integrated_chatbot integrated_chatbot_node.py
    or
   roslaunch integrated_chatbot integrated_chatbot_solid.launch
or 
roslaunch integrated_chatbot integrated_chatbot_solid.launch  ( to run both camera and chatbot (need more capabilities for more performance)


    ```

2. **Interact using voice** after your face is recognized.
   - Known users are greeted personally.
   - You can speak to the bot in Arabic.
   - Say "شكرا" or "مع السلامة" to end the conversation.

---


## 🧪 Gemini AI Integration

Gemini is used for general Arabic Q&A. Prompts are structured to ensure:

- Fluent and correct Modern Standard Arabic (MSA)
- No special symbols or markdown
- Polite responses or clarification requests if the question is unclear

---

## 🎥 Live Feed

Live camera feed with bounding boxes:
- Green: Recognized person
- Red: Unknown
- Labels include **name, gender, and emotion**

Press `q` to quit the live window.

---

> This refactored version **preserves all original functionality** while offering **superior maintainability, extensibility, and clarity**.

## 🧱 Refactored Modular Architecture

The chatbot has been **refactored** into a clean and maintainable modular structure:

### 📁 Project Structure

```
integrated_chatbot/
├── scripts/
│   ├── main.py                  # Main entry point (ROS node)
│   ├── face_recognizer.py       # Face detection, recognition, emotion & gender analysis
│   ├── voice_interface.py       # Voice recognition and speech synthesis
│   ├── gemini_handler.py        # Gemini API prompt generation and response parsing
│   └── conversation_manager.py  # Manages conversational state and interactions
├── launch/
│   └── integrated_chatbot_solid.launch
├── models/                      # ONNX/Caffe models for emotion/gender detection
├── data/                        # Folder containing known face images
├── CMakeLists.txt
└── package.xml
```

### 🔧 Design Improvements

#### ✅ Modular Architecture:
- Functionality split into classes:
  - `FaceRecognizer`
  - `VoiceInterface`
  - `GeminiHandler`
  - `ConversationManager`
- Each class handles a single, well-defined responsibility

#### ✅ State Management:
- Uses `Enum` for clear state transitions
- Better encapsulation of dialogue states

#### ✅ Error Handling:
- Consistent try-except blocks
- Separation of user-related, hardware, and API errors

#### ✅ Code Organization:
- Private methods prefixed with `_`
- Logical method grouping
- Clean, consistent naming

#### ✅ Configuration Management:
- Centralized config loading from `rosparam`
- Easy to update parameters without touching logic

#### ✅ Reduced Duplication:
- Common logic extracted to helper methods
- DRY (Don't Repeat Yourself) principles applied

#### ✅ Improved Readability:
- Clear method and variable names
- Grouped operations by function
- Fully commented logic where needed


---
for vision_chatbot_pub_sub 



## 📦 Project Structure

```bash
vision_chatbot/
├── CMakeLists.txt
├── package.xml
├── launch/
│   ├── vision_chatbot.launch
│   └── vision_chatbot_solid.launch
├── models/
│   ├── emotion-ferplus-8.onnx
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── data/                     # Known face images
├── scripts/
│   └── vision_chatbot_node.py
├── src/
├── config/
└── README.md


---


# Navigate to your workspace
cd ~/catkin_ws

# Build your package
catkin_make

# Source your workspace
source devel/setup.bash

# Run the node
roslaunch vision_chatbot vision_chatbot.launch
----
