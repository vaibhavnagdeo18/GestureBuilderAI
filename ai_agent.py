# ai_agent.py
import os
import json
from openai import OpenAI

# -----------------------------
# 🔑 Initialize OpenAI client safely
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("❌ Missing OpenAI API key. Please set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=api_key)


# -----------------------------
# 🧠 Function: Interpret Command
# -----------------------------
def interpret_command(command: str):
    """
    Interprets a natural language command into structured 3D actions.

    Example:
        Input: "Add a big red cube above the ground"
        Output:
        {
            "action": "create",
            "target": "cube",
            "color": "red",
            "position": [0, 1, 0],
            "amount": 1.0
        }
    """
    if not command.strip():
        print("⚠️ Empty command received — skipping AI interpretation.")
        return None

    system_prompt = """
    You are a 3D world assistant.
    Convert user commands into **structured JSON** describing the 3D scene changes.

    Respond ONLY in valid JSON with this schema:
    {
        "action": "create" | "delete" | "scale" | "rotate" | "move" | "color",
        "target": "cube" | "sphere" | "pyramid" | "tree" | "ground" | "any",
        "color": "red" | "blue" | "green" | "yellow" | "white" | "black" | null,
        "position": [x, y, z] | null,
        "amount": float | null
    }

    Rules:
    - If the command lacks a value (like color or position), set it to null.
    - Never include text or explanation outside of the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": command},
            ],
            response_format={"type": "json_object"},  # Strict JSON mode
        )

        raw_output = response.choices[0].message.content
        print(f"🤖 Raw AI Output: {raw_output}")

        parsed = json.loads(raw_output)
        print(f"✅ Parsed AI Output: {json.dumps(parsed, indent=2)}")

        return parsed

    except json.JSONDecodeError:
        print("⚠️ JSON decoding failed — invalid model output.")
        print("Raw output:", raw_output)
        return None

    except Exception as e:
        print(f"❌ Error interpreting command: {e}")
        return None