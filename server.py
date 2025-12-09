from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import threading
import json
from gesture_recognizer import get_gesture, run_gesture_loop
from ai_agent import interpret_command  # ✅ use the real one now

app = FastAPI(title="Gesture World Builder API", version="1.0")

# 🌍 Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_ai_command = None  # ✅ memory to store last AI result


@app.get("/")
def home():
    return {"message": "✨ Gesture World Builder Backend is running!"}


@app.get("/status")
def status():
    return {"status": "ok", "server": "running", "port": 8085}


# ✋ Gesture Recognition
@app.get("/gesture")
def read_gesture():
    gesture = get_gesture() or "NONE"
    print(f"[SERVER] Latest gesture detected: {gesture}")

    gesture_map = {
        "PINCH": "cube",
        "OPEN_PALM": "sphere",
        "FIST": "tree",
        "SWIPE_LEFT": "rotate_left",
        "SWIPE_RIGHT": "rotate_right",
    }

    shape = gesture_map.get(gesture, "none")
    print(f"[SERVER] Sending → Gesture: {gesture}, Shape: {shape}")
    return JSONResponse(content={"gesture": gesture, "shape": shape})


# 💬 POST: receive new AI prompt
@app.post("/ai-command")
async def ai_command(data: dict):
    global latest_ai_command
    prompt = data.get("prompt", "").strip()
    print(f"[SERVER] 🧠 Received AI command prompt: {prompt}")

    if not prompt:
        return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

    try:
        # 🧠 Real AI interpretation
        ai_result = interpret_command(prompt)
        print(f"[SERVER] ✅ AI interpreted result:\n{json.dumps(ai_result, indent=2)}")

        # Store result globally for polling
        latest_ai_command = ai_result

        return JSONResponse(content={
            "response": "✅ Command processed successfully!",
            "data": ai_result
        })

    except Exception as e:
        print("❌ Error processing AI command:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 💬 GET: allow main.py to fetch latest AI result
@app.get("/ai-command")
def get_latest_ai_command():
    global latest_ai_command
    if latest_ai_command:
        return JSONResponse(content={"ai_response": latest_ai_command})
    return JSONResponse(content={"ai_response": None})


# 🚀 Run Gesture Recognition + API
if __name__ == "__main__":
    gesture_thread = threading.Thread(target=run_gesture_loop, daemon=True)
    gesture_thread.start()

    print("✅ Gesture recognition thread started.")
    print("🌐 API running at: http://localhost:8085")
    print("📡 Waiting for gesture or AI prompt input...")

    uvicorn.run(app, host="0.0.0.0", port=8085)