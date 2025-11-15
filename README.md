**SET-Visual-Player**

SET-AI-Agent is a real-time, AI-powered web application that lets you play the SET card game using live camera input.
It combines computer vision, deep learning, and web technologies to recognize cards, detect valid sets, and enable interactive gameplay directly from your browser.

**Overview**

The project bridges real-time visual recognition and human gameplay through a multi-component architecture:

A CNN-based vision models that relies on two Convolutional Neural Networks **YOLOv11n-seg** for **Card segmentation** and **MobileNetV4** for **Card classification**.

A FastAPI backend handles model inference, board state management, and set validation.

A React frontend provides a responsive, intuitive interface for live gameplay and visualization.

The result: a seamless experience that merges classical game logic with modern computer vision techniques.

**Key Features**

- Real-time hand detection using OpenCV 

- Fully-trained Augmented MobilenetV4 for card classification (4 linear heads for each attribute)

- YOLOv11n-seg for card Segmentation using a binary mask for each pixel.

- smart card extraction from mask using CV2.

- Intelligent board tracking with temporal confidence logic.

- FastAPI backend serving inference and game logic (board and AI \ Human Player classes) using WS.

- React web interface for live camera streaming and user interaction.

- Set validation engine ensuring correctness and speed

- Modular design for easy extension and research experimentation


**Installation & Setup**

_1. Clone the Repository_
```
git clone https://github.com/TheBananaBender/Set_vision_player
cd Set_vision_player
```

_2. Backend Setup_
```text
cd backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

_This launches the FastAPI server locally on http://127.0.0.1:8000._

_3. Frontend Setup_

In a separate terminal:

```text
cd set-vision-web
npm install
npm run build
npm start
```

Your local web interface should now be running on http://localhost:3000
 and communicating with the backend.

**Tech Stack**

Python's OOP for Players and Board design 
Frontend - React, JavaScript, CSS
Backend	FastAPI, Uvicorn
Computer Vision	OpenCV, PyTorch
Build Tools	npm

**Future Improvements**

- Improve model robustness with larger card datasets

- Improve temporal confidence logic of hands && cards

- Implementing end_of_game logic when (grave_yard_count + board_cards == 81 && !board.has_set)

- Deploy full system via Docker or cloud containerization

🏗️ Project Architecture

```text
SET_Visual_Player/
├── backend/
│   ├── main.py
│   ├── Online_game_proto.py        # no - website working demo
│   ├── utils.py
│   ├── Game_logic/
│   │   └── Set_game_mechanics.py
│   ├── Players/
│   │   ├── Human_agent.py
│   │   └── vision_agent.py
│   └── vision_models/
│       ├── vision_models.py
│       ├── SET_yolo_model/
│       │   ├── train.py
│       │   └── best.pt
│       └── handtest.py
├── set-vision-web/
│   ├── package.json
│   ├── vite.config.js
│   ├── dev_run_backend.py
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── styles.css
│       ├── assets/
│       │   └── set/
│       └── components/
│           ├── WebcamFeed.jsx
│           └── ...
```

**Requirements**

_Backend_

- Python >= 3.10

- pip (latest)

- optional; virtualenv or venv for isolation 

- GPU optional; for CUDA builds install PyTorch with matching CUDA toolkit

- Python packages: fastapi, uvicorn, opencv-python, numpy, torch, torchvision, ultralytics, timm, mediapipe

_Frontend_

- Node.js >= 18

- npm >= 9 (bundled with Node)

- Modern browser with WebRTC access for the webcam feed


**Authors:**

- **Roy Dahan** => [TheBananaBender](https://github.com/TheBananaBender/) 

- **Gal Salman** => [GalSal1](https://github.com/Galsal1/)

