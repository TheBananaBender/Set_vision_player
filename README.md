SET-AI-Agent

SET-AI-Agent is a real-time, AI-powered web application that lets you play the SET card game using live camera input.
It combines computer vision, deep learning, and web technologies to recognize cards, detect valid sets, and enable interactive gameplay directly from your browser.

> **Overview**

The project bridges real-time visual recognition and human gameplay through a multi-component architecture:

A CNN-based vision model predicts card polygons and attributes from camera frames.

A FastAPI backend handles model inference, board state management, and set validation.

A React frontend provides a responsive, intuitive interface for live gameplay and visualization.

The result: a seamless experience that merges classical game logic with modern computer vision techniques.

🧩 Key Features

- Real-time hand detection using OpenCV



- Intelligent board tracking with temporal confidence logic

- FastAPI backend serving inference and game logic endpoints

- React web interface for live camera streaming and user interaction

- Set validation engine ensuring correctness and speed

- Modular design for easy extension and research experimentation


⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/SET-AI-Agent.git
cd SET-AI-Agent
2. Backend Setup
cd backend
uvicorn main:app --reload

_This launches the FastAPI server locally on http://127.0.0.1:8000
._

3. Frontend Setup

In a separate terminal:

cd set-vision-web
npm install
npm run build
npm start


Your local web interface should now be running on http://localhost:3000
 and communicating with the backend.

Tech Stack
Python's OOP for Players and Board design 
Frontend - React, JavaScript, CSS
Backend	FastAPI, Uvicorn
Computer Vision	OpenCV, PyTorch
Build Tools	npm
🧪 Future Improvements

- Improve model robustness with larger card datasets

- Add player score tracking and multiplayer support

- Deploy full system via Docker or cloud containerization

- Add mobile device camera support

- Authors

- 🧠 SET-AI-Agent

SET-AI-Agent is a real-time, AI-powered web application that lets you play the SET card game using live camera input.
It combines computer vision, deep learning, and web technologies to recognize cards, detect valid sets, and enable interactive gameplay directly from your browser.

🚀 Overview

The project bridges real-time visual recognition and human gameplay through a multi-component architecture:

A CNN-based vision model predicts card polygons and attributes from camera frames.

A FastAPI backend handles model inference, board state management, and set validation.

A React frontend provides a responsive, intuitive interface for live gameplay and visualization.

The result: a seamless experience that merges classical game logic with modern computer vision techniques.

🧩 Key Features

🎥 Real-time card detection via OpenCV + PyTorch

🧮 Intelligent board tracking with temporal confidence logic

⚡ FastAPI backend serving inference and game logic endpoints

🌐 React web interface for live camera streaming and user interaction

🧠 Set validation engine ensuring correctness and speed

🧰 Modular design for easy extension and research experimentation

🏗️ Project Architecture
SET-AI-Agent/
├── backend/                 # FastAPI + PyTorch inference server
│   ├── models/              # Trained CNN model and weights
│   ├── utils/               # Image processing and board management
│   └── main.py              # Entry point for the backend
│
├── set-vision-web/          # React frontend
│   ├── src/                 # Components, hooks, and UI logic
│   ├── public/              # Static assets
│   └── package.json         # Frontend dependencies
│
└── README.md                # (this file)

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/SET-AI-Agent.git
cd SET-AI-Agent

2. Backend Setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload


This launches the FastAPI server locally on http://127.0.0.1:8000
.

3. Frontend Setup

In a separate terminal:

cd set-vision-web
npm install
npm run build
npm start


Your local web interface should now be running on http://localhost:3000
 and communicating with the backend.

🧠 Tech Stack
Layer	Technologies
Frontend	React, JavaScript, CSS
Backend	FastAPI, Uvicorn
Computer Vision	OpenCV, PyTorch
Language	Python
Build Tools	npm
🧪 Future Improvements

🧬 Improve model robustness with larger card datasets

🎯 Add player score tracking and multiplayer support

💡 Deploy full system via Docker or cloud containerization

📱 Add mobile device camera support

🖋️ Author

[ Roy Dahan => TheBananaBender && Gal Salman => GalSal1 ] 
  

[Your Name]
📍 Computer Science Student at TAU & TUM
