from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# === Setup Directories ===
REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
FEEDBACK_FILE = os.path.join(os.getcwd(), "feedback_log.json")

os.makedirs(REFERENCE_DIR, exist_ok=True)

print(f"📁 Reference directory: {REFERENCE_DIR}")
if not os.listdir(REFERENCE_DIR):
    print("⚠️  No reference files found. You can upload or sync patterns later.")
else:
    print(f"📁 Files found: {os.listdir(REFERENCE_DIR)}")

# === Initialize App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten to Base44 domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Health Check ===
@app.get("/health")
def health():
    return {"status": "ok"}

# === Image Comparison Logic ===
def compare_images(img1, img2):
    """Compare two images and return similarity score."""
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

# === Analyze Chart Function ===
async def analyze_uploaded_chart(image_path: str):
    """Analyze uploaded chart and find the closest match from reference library."""
    try:
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
                continue  # ✅ Skip comparing the file to itself
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue

            score = compare_images(uploaded_img, ref_img)
            if score > best_score:
                best_score = score
                best_match = ref_file

        if best_match:
            return {
                "status": "success",
                "best_match": best_match,
                "confidence_score": round(float(best_score), 3),
            }
        else:
            return {"status": "no_match", "message": "No suitable match found."}

    except Exception as e:
        print("❌ Analyze chart error:", e)
        return {"status": "error", "message": str(e)}

# === Main Upload + Analyze Endpoint ===
@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    """
    Upload a chart, save it to the reference folder, and automatically analyze it.
    """
    try:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        save_path = os.path.join(REFERENCE_DIR, file.filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Saved new chart: {save_path}")

        # Run automatic analysis
        analysis_result = await analyze_uploaded_chart(save_path)

        response = {
            "status": "success",
            "message": f"Chart '{file.filename}' uploaded and analyzed successfully.",
            "uploaded_chart_url": f"{save_path}",
            "analysis_result": analysis_result,
        }

        return JSONResponse(
            content=response,
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

    except Exception as e:
        print(f"❌ Error in /process_chart: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

# === Feedback Endpoint ===
@app.post("/feedback")
async def feedback(request: Request):
    """
    Store user feedback about chart matches.
    """
    try:
        data = await request.json()
        feedback_entry = {
            "uploaded_chart": data.get("uploaded_chart"),
            "matched_chart": data.get("matched_chart"),
            "confidence": data.get("confidence"),
            "is_correct": data.get("is_correct"),
            "timestamp": datetime.utcnow().isoformat()
        }

        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback_data = json.load(f)

        feedback_data.append(feedback_entry)

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_data, f, indent=2)

        print(f"🧠 Feedback recorded: {feedback_entry}")

        return JSONResponse(
            content={"status": "success", "message": "Feedback recorded."},
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

    except Exception as e:
        print("❌ Feedback error:", str(e))
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

# === Root Endpoint ===
@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

# === Run Server ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
