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

# ============================================================
# 🧱 SETUP
# ============================================================

# Define reference chart storage directory
REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
os.makedirs(REFERENCE_DIR, exist_ok=True)

print(f"📁 Reference directory: {REFERENCE_DIR}")
if not os.listdir(REFERENCE_DIR):
    print("⚠️  No reference files found. You can upload or sync patterns later.")
else:
    print(f"📁 Files found: {os.listdir(REFERENCE_DIR)}")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Base44 and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Base44 URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🌡️ HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# 💾 SIMPLE UPLOAD (optional legacy route)
# ============================================================

@app.post("/upsert")
async def upsert_chart(file: UploadFile = File(...)):
    """Simple upload for testing direct chart storage"""
    try:
        contents = await file.read()
        upload_path = os.path.join("uploaded_charts", file.filename)
        os.makedirs("uploaded_charts", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(contents)

        print(f"✅ Uploaded: {file.filename} ({len(contents)} bytes)")
        return JSONResponse(
            content={
                "status": "success",
                "message": "Chart stored successfully.",
                "filename": file.filename,
            },
            status_code=200,
        )
    except Exception as e:
        print("❌ Upsert error:", str(e))
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )

# ============================================================
# 🧩 IMAGE COMPARISON LOGIC
# ============================================================

def compare_images(img1, img2):
    """Compute similarity between two images"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
    img1_gray = cv2.resize(img1_gray, (w, h))
    img2_gray = cv2.resize(img2_gray, (w, h))

    score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(score)
    return float(max_val)

# ============================================================
# 🧠 ANALYZE UPLOADED CHART AGAINST REFERENCES
# ============================================================

async def analyze_uploaded_chart(image_path: str):
    """Compare one uploaded image against stored reference charts."""
    try:
        reference_files = [
            f for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not reference_files:
            return {"status": "no_references", "message": "No reference charts found."}

        uploaded_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if uploaded_img is None:
            return {"status": "error", "message": "Invalid uploaded image."}

        best_match = None
        best_score = -1

        for ref_file in reference_files:
            ref_path = os.path.join(REFERENCE_DIR, ref_file)
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                continue

            ref_img_resized = cv2.resize(ref_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
            score = cv2.matchTemplate(uploaded_img, ref_img_resized, cv2.TM_CCOEFF_NORMED).max()

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
        return {"status": "error", "message": str(e)}

# ============================================================
# 📤 MANUAL REFERENCE UPLOAD
# ============================================================

@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a chart manually to reference library."""
    try:
        save_path = os.path.join(REFERENCE_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📥 Reference chart saved: {save_path}")
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Reference '{file.filename}' uploaded successfully.",
                "path": save_path,
            },
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )

# ============================================================
# 🔄 PROCESS CHART (AUTO UPLOAD + ANALYZE)
# ============================================================

@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    """
    Upload a chart, save it to the reference folder,
    and automatically analyze it against stored patterns.
    """
    try:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        save_path = os.path.join(REFERENCE_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Saved new chart: {save_path}")

        # Run analysis immediately
        analysis_result = await analyze_uploaded_chart(save_path)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Chart '{file.filename}' uploaded and analyzed successfully.",
                "file_path": save_path,
                "analysis_result": analysis_result,
            },
            status_code=200,
        )

    except Exception as e:
        print(f"❌ Error in /process_chart: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )

# ============================================================
# 🚀 ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

# ============================================================
# 🏁 ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
