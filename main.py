from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np
import cv2
import io
import os
import glob
import shutil
import requests

# ====================================
# 🌍 DIRECTORY SETUP
# ====================================
BASE_DIR = os.getcwd()
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_patterns")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_charts")

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ====================================
# 🚀 FASTAPI INITIALIZATION
# ====================================
app = FastAPI(title="TradeMirror Chart Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace with Base44 domain later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================
# 🧠 HELPER FUNCTIONS
# ====================================
def compare_images(img1, img2):
    """Compare two images and return a similarity score."""
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Match resolution
        h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
        img1_gray = cv2.resize(img1_gray, (w, h))
        img2_gray = cv2.resize(img2_gray, (w, h))

        # Structural correlation
        score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(score)
        return float(max_val)
    except Exception:
        return 0.0


def get_public_url(filename: str, folder: str):
    """Generate a public Render URL for an image."""
    safe_name = filename.replace(" ", "_")
    return f"https://chart-backend-ht00.onrender.com/{folder}/{safe_name}"


# ====================================
# 🩺 HEALTH & STATIC ROUTES
# ====================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"status": "Backend running successfully"}


@app.get("/reference/{filename}")
def get_reference_image(filename: str):
    """Serve reference chart images."""
    file_path = os.path.join(REFERENCE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"status": "error", "message": "Reference file not found"}, status_code=404)


@app.get("/uploaded/{filename}")
def get_uploaded_image(filename: str):
    """Serve uploaded chart images."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"status": "error", "message": "Uploaded file not found"}, status_code=404)


# ====================================
# 🔍 ANALYSIS LOGIC
# ====================================
async def analyze_uploaded_chart(image_path: str):
    """Compare uploaded chart against all stored reference patterns, excluding itself."""
    try:
        reference_files = [
            f for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not reference_files:
            return {"status": "no_references", "message": "No reference charts found."}

        uploaded_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if uploaded_img is None:
            return {"status": "error", "message": "Invalid uploaded image."}

        uploaded_filename = os.path.basename(image_path)
        results = []

        for ref_file in reference_files:
            # ✅ Skip comparing the file against itself
            if ref_file == uploaded_filename:
                continue

            ref_path = os.path.join(REFERENCE_DIR, ref_file)
            ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            if ref_img is None:
                continue

            score = compare_images(uploaded_img, ref_img)
            results.append({
                "reference": ref_file,
                "similarity": round(score, 3),
                "reference_url": get_public_url(ref_file, "reference")
            })

        if not results:
            return {
                "status": "no_match",
                "message": "No other charts available for comparison yet."
            }

        # Sort by similarity score
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        best_match = results[0]

        return {
            "status": "success",
            "best_match": best_match["reference"],
            "best_match_url": best_match["reference_url"],
            "confidence_score": best_match["similarity"],
            "top_matches": results[:3],
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ====================================
# 📤 UPLOAD + AUTO-LEARN
# ====================================
@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    """
    Upload a chart → Save to uploads + reference folder →
    Run analysis → Return results and image URLs.
    """
    try:
        file_name = file.filename.replace(" ", "_")

        # Save in /uploaded_charts
        upload_path = os.path.join(UPLOAD_DIR, file_name)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Saved new chart in uploads: {upload_path}")

        # Add to /reference_patterns for ML library
        reference_path = os.path.join(REFERENCE_DIR, file_name)
        shutil.copy(upload_path, reference_path)
        print(f"📚 Added to reference library: {reference_path}")

        # Analyze against previous charts (excluding itself)
        analysis = await analyze_uploaded_chart(upload_path)

        response = {
            "status": "success",
            "message": f"Chart '{file_name}' uploaded, added to library, and analyzed successfully.",
            "uploaded_chart_url": get_public_url(file_name, "uploaded"),
            "analysis_result": analysis,
        }

        return JSONResponse(response, status_code=200)

    except Exception as e:
        print(f"❌ Error in /process_chart: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ====================================
# 🧱 SERVER STARTUP
# ====================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
