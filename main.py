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

# --- Paths ---
BASE_DIR = os.getcwd()
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_patterns")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_charts")

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- App Setup ---
app = FastAPI(title="Chart Pattern Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for testing; replace later with your Base44 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---
def compare_images(img1, img2):
    """Compare two charts using structural similarity."""
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
        img1_gray = cv2.resize(img1_gray, (w, h))
        img2_gray = cv2.resize(img2_gray, (w, h))

        # Template matching for visual similarity
        score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(score)
        return float(max_val)
    except Exception:
        return 0.0

def get_public_url(filename: str, folder: str):
    """Return a public-style URL for an image."""
    safe_name = filename.replace(" ", "_")
    return f"https://chart-backend-ht00.onrender.com/{folder}/{safe_name}"

# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "Backend running successfully"}

@app.get("/reference/{filename}")
def get_reference_image(filename: str):
    """Serve reference pattern images."""
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

# --- Analyze Function ---
async def analyze_uploaded_chart(image_path: str):
    """Compare an uploaded chart with all reference patterns."""
    reference_files = [
        f for f in os.listdir(REFERENCE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not reference_files:
        return {"status": "no_references", "message": "No reference charts found."}

    uploaded_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if uploaded_img is None:
        return {"status": "error", "message": "Invalid uploaded image."}

    best_match = None
    best_score = -1

    for ref_file in reference_files:
        ref_path = os.path.join(REFERENCE_DIR, ref_file)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
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
            "confidence_score": round(best_score, 3),
            "best_match_url": get_public_url(best_match, "reference"),
        }
    else:
        return {"status": "no_match", "message": "No match found."}

# --- Core Upload + Auto-Analyze ---
@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    """
    Uploads a chart, stores it, analyzes it against reference patterns,
    and returns URLs for both uploaded and best-match charts.
    """
    try:
        # Save the uploaded chart
        file_name = file.filename.replace(" ", "_")
        save_path = os.path.join(UPLOAD_DIR, file_name)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Saved new chart: {save_path}")

        # Analyze the chart
        analysis = await analyze_uploaded_chart(save_path)

        # Compose result
        response = {
            "status": "success",
            "message": f"Chart '{file_name}' uploaded and analyzed successfully.",
            "uploaded_chart_url": get_public_url(file_name, "uploaded"),
            "analysis_result": analysis,
        }

        return JSONResponse(response, status_code=200)

    except Exception as e:
        print(f"❌ Error in /process_chart: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# --- Server Start ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
