# demo/run_demo.py
import time
import random
import os

if os.getenv("CI") == "true":
    print("Skipping real demo in CI environment.")
    exit(0)

def fake_inference():
    gestures = ["thumbs_up", "thumbs_down", "swipe_left", "swipe_right", "fist"]
    return random.choice(gestures), round(random.random(), 3)

def main(n=5):
    print("GestureBuilderAI — minimal demo")
    for i in range(n):
        g, c = fake_inference()
        print(f"[{i+1}] prediction={g:12} confidence={c}")
        time.sleep(0.3)
    print("Demo finished (placeholder). Replace with real inference call to run full system.")

if __name__ == "__main__":
    main()