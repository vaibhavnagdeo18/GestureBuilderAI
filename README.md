# GestureBuilderAI — Real-time Gesture Builder & Inference (Showcase)

A real-time gesture creation and inference system built using Python + MediaPipe + lightweight ML models.  
Designed for prototyping gesture-based interaction systems, dataset collection, training, and live inference.

## 🚀 Demo (quick)

Run the minimal demo:
```bash

python demo/run_demo.py

NOTE: MediaPipe is known not to support Python 3.12/3.13 in many environments. Use Python 3.10 for best compatibility.

## ⚡ Quick Start (30s)

1. Clone:
```bash
git clone https://github.com/vaibhavnagdeo18/GestureBuilderAI.git
cd GestureBuilderAI

2.	Create venv (recommended Python 3.10):
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

3.	Run the demo:
python demo/run_demo.py


## Key Features

- Real-time gesture tracking & landmark extraction (MediaPipe/OpenCV)  
- Gesture dataset builder and recording utilities  
- Lightweight model training & export pipeline (TensorFlow/PyTorch compatible)  
- Modular inference engine and agent integration (`ai_agent.py`, `gesture_recognizer.py`)  
- Extensible state machine / world manager for interactions

## Project structure (top-level)

gesture-world-builder/
│
├── ai_agent.py            # Core AI agent logic
├── gesture_recog.py       # Gesture recognition pipeline
├── main.py                # Main runner / integration layer
├── server.py              # Optional server support
├── renderer.py            # Visualization / UI
├── world_manager.py       # World / state logic
├── world_state.json       # Global world state config
│
├── demo/
│   ├── run_demo.py
│   └── assets/
│
├── requirements.txt
├── .gitignore
└── README.md

---

## My Contribution

I designed and implemented the full pipeline: data collection, landmark extraction, model training + inference, and a demo/renderer for interactive testing. This repo is my cleaned, reviewable showcase.

## Contributing
PRs are welcome. Start with docs, tests, or adding gesture examples.

## Contact
Vaibhav Nagdeo — vaibhavnagdeo@gmail.com — GitHub: @vaibhavnagdeo18
