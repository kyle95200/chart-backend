from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import io
import os
import glob
import shutil
import requests
import json
from datetime import datetime

# === Directories ===
REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
FEEDBACK_FILE = os.path.join(os.getcwd(), "feedback_log.json")
MEMORY_FILE = os.path.join(os.getcwd(), "pattern_memory.json")

os.makedirs(REFERENCE_DIR, exist_ok=True)
for f in [FEEDBACK_FILE, MEMORY_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as file:
            if "feedback" in f:
                json.dump([], file)
            else:
                json.dump({}, file)

# === Initialize App ===
app = FastAPI()
app.mount("/reference_patterns", StaticFiles(directory=REFERENCE_DIR), name="reference_patterns")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Base44 frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# === Core Comparison ===
def compare_images(img1, img2):
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
        img1_gray = cv2.resize(img1_gray, (w, h))
        img2_gray = cv2.resize(img2_gray, (w, h))
        score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(score)
        return float(max_val)
    except Exception as e:
        print("❌ Image comparison error:", e)
        return 0.0

# === Adaptive Memory ===
def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def update_pattern_memory(match_name, confidence, correct):
    memory = load_memory()
    if match_name not in memory:
        memory[match_name] = {"total": 0, "correct": 0, "score": 0}

    memory[match_name]["total"] += 1
    if correct:
        memory[match_name]["correct"] += 1

    accuracy = memory[match_name]["correct"] / memory[match_name]["total"]
    memory[match_name]["score"] = round((accuracy + confidence) / 2, 3)
    save_memory(memory)
    print(f"🧩 Memory updated for {match_name}: {memory[match_name]}")

# === Chart Analysis ===
async def analyze_uploaded_chart(image_path: str):
    try:
        memory = load_memory()
        reference_files = [
            f for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not reference_files:
            return {"status": "no_references", "message": "No reference charts found."}

        uploaded_img = cv2.imread(image_path)
        if uploaded_img is None:
            return {"status": "error", "message": "Invalid uploaded image."}

        best_match = None
        best_score = -1

        for ref_file in reference_files:
            ref_path = os.path.join(REFERENCE_DIR, ref_file)
            if os.path.abspath(ref_path) == os.path.abspath(image_path):
                continue

            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue

            base_score = compare_images(uploaded_img, ref_img)
            learned_weight = memory.get(ref_file, {}).get("score", 0.5)
            weighted_score = (base_score + learned_weight) / 2

            if weighted_score > best_score:
                best_score = weighted_score
                best_match = ref_file

        if best_match:
            match_url = f"https://chart-backend-ht00.onrender.com/reference_patterns/{best_match}"
            return {
                "status": "success",
                "best_match": best_match,
                "best_match_url": match_url,
                "confidence_score": round(float(best_score), 3),
            }
        else:
            return {"status": "no_match", "message": "No suitable match found."}

    except Exception as e:
        print("❌ Analyze chart error:", e)
        return {"status": "error", "message": str(e)}

# === Upload + Auto Analyze ===
@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    try:
        save_path = os.path.join(REFERENCE_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Saved new chart: {save_path}")

        public_url = f"https://chart-backend-ht00.onrender.com/reference_patterns/{os.path.basename(save_path)}"
        analysis_result = await analyze_uploaded_chart(save_path)

        response = {
            "status": "success",
            "message": f"Chart '{file.filename}' uploaded and analyzed successfully.",
            "uploaded_chart_url": public_url,
            "analysis_result": analysis_result,
        }
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        print(f"❌ Error in /process_chart: {str(e)}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# === Feedback Route (Fixed) ===
@app.post("/feedback")
async def feedback(request: Request):
    try:
        data = await request.json()
        feedback_entry = {
            "uploaded_chart": data.get("uploaded_chart"),
            "matched_chart": data.get("matched_chart"),
            "confidence": data.get("confidence"),
            "is_correct": data.get("is_correct"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                try:
                    content = json.load(f)
                    if isinstance(content, list):
                        feedback_data = content
                    elif isinstance(content, dict):
                        feedback_data = [content]
                except json.JSONDecodeError:
                    feedback_data = []

        feedback_data.append(feedback_entry)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_data, f, indent=2)

        if feedback_entry["matched_chart"]:
            match_name = os.path.basename(feedback_entry["matched_chart"])
            update_pattern_memory(match_name, feedback_entry["confidence"], feedback_entry["is_correct"])

        print(f"🧠 Feedback recorded and memory updated: {feedback_entry}")
        return JSONResponse(content={"status": "success", "message": "Feedback recorded and learned."}, status_code=200)

    except Exception as e:
        print("❌ Feedback error:", str(e))
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# === AI Summary ===
@app.get("/ai_summary")
def ai_summary():
    """
    Returns a performance summary of AI learning progress.
    Includes total patterns trained, average confidence, and top performers.
    """
    try:
        if not os.path.exists(MEMORY_FILE):
            return {"status": "error", "message": "No memory file found."}

        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)

        if not memory:
            return {"status": "no_data", "message": "No patterns trained yet."}

        total_patterns = len(memory)
        total_feedback = sum(m.get("total", 0) for m in memory.values())
        avg_accuracy = (
            sum(m.get("correct", 0) / m.get("total", 1) for m in memory.values()) / total_patterns
        )
        avg_score = sum(m.get("score", 0) for m in memory.values()) / total_patterns

        top_patterns = sorted(memory.items(), key=lambda x: x[1].get("score", 0), reverse=True)[:3]

        return {
            "status": "success",
            "summary": {
                "total_patterns": total_patterns,
                "total_feedback": total_feedback,
                "average_accuracy": round(avg_accuracy * 100, 2),
                "average_confidence": round(avg_score * 100, 2),
                "top_patterns": [
                    {"name": name, **data} for name, data in top_patterns
                ],
            },
        }

    except Exception as e:
        print("❌ AI summary error:", e)
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

# === Start Server ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
