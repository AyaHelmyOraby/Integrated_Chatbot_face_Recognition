
# ROS Workspace Structure Overview

This repository contains several ROS packages developed for face recognition, chatbot integration, and their combined use. Below is a detailed description of the folder structure and launch files.

---

## 📁 Folder Structure

```
catkin_ws/
├── build/
├── devel/
├── install/
├── logs/
├── src/
│   ├── turtlebot3_face_recognition/
│   ├── voice_chatbot/
│   ├── integrated_chatbot/
│   ├── vision_chatbot/
│   ├── ...
```

---

## 🔹 Package Descriptions

### 🧠 turtlebot3_face_recognition
- **Purpose:** Face recognition only.
- **Launch Files:**
  - `camera_face_recognizer.launch` – Launch with camera for real-time recognition.
  - `laptop_cam_face.launch` – Use laptop camera.
  - `face_detection_image.launch` – Run face detection on static images.
  - `face_detection_video.launch` – Run face detection on video files.
  - `laptop_cam_face_recognizer.launch` – Combined face detection and recognition using the laptop camera.

### 🗣️ voice_chatbot
- **Purpose:** Chatbot (voice) interaction only.

### 🔄 integrated_chatbot
- **Purpose:** Integrates both face recognition and chatbot.
- **Launch Files:**
  - `integrated_chatbot.launch` – All functionality in one file.
  - `integrated_chatbot_solid.launch` – Applies SOLID principles:
    - Separates functions into files.
    - Chatbot activates **after** face recognition.
    - Easier to edit and maintain.
  - `integrated_chatbot_solid_together.launch` – Launches camera and chatbot at the same time (may reduce performance due to higher power consumption).

### 👁️ vision_chatbot
- **Purpose:** Applies publisher/subscriber concepts clearly (as explained in the report).
- **Launch Files:**
  - `vision_chatbot.launch` – All functions in a single file.
  - `vision_chatbot_solid.launch` – Functions separated into files for easier editing and refactoring.


---

## ⚙️ Environment Setup and Running Instructions

Before running any launch file, make sure to configure the environment properly.

### ✅ 1. Source the Catkin Workspace

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

If using a virtual environment, activate it:

```bash
source path/to/your/venv/bin/activate
```

If you're using a specific Python environment like `noor_venv_py39_clean`, activate it:

```bash
conda activate noor_venv_py39_clean
# or
source ~/path/to/noor_venv_py39_clean/bin/activate
```

### ✅ 2. Run Launch Files

Make sure you're in the correct workspace and sourced before running:

```bash
roslaunch <package_name> <launch_file.launch>
```

Examples:

```bash
roslaunch turtlebot3_face_recognition camera_face_recognizer.launch
roslaunch voice_chatbot chatbot.launch
roslaunch integrated_chatbot integrated_chatbot_solid.launch
roslaunch vision_chatbot vision_chatbot.launch
```

Make sure `roscore` is running in a separate terminal if not managed by the launch file.

---
