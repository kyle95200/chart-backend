from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import io
import os
import glob
from fastapi import FastAPI, File, UploadFile, Request

# Ensure reference_patterns directory exists
REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Log what’s inside
print(f"📁 Reference directory: {REFERENCE_DIR}")
if not os.listdir(REFERENCE_DIR):
    print("⚠️  No reference files found. You can upload or sync patterns later.")
else:
    print(f"📁 Files found: {os.listdir(REFERENCE_DIR)}")

app = FastAPI()

# Allow Base44 or other frontend domains to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Base44 app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upsert")
async def upsert_chart(file: UploadFile = File(...)):
    """
    Receives a chart upload from Base44 and stores or updates it.
    """
    try:
        # Read uploaded file
        contents = await file.read()

        # Log for debugging
        print(f"Received file: {file.filename}, size={len(contents)} bytes")

        # Save uploaded file locally (optional)
        upload_path = os.path.join("uploaded_charts", file.filename)
        os.makedirs("uploaded_charts", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(contents)

        # Here you can update your vector DB or trigger analysis, etc.
        print(f"Chart saved at: {upload_path}")

        return {
            "status": "success",
            "message": "Chart received and stored successfully.",
            "filename": file.filename
        }

    except Exception as e:
        print("Upsert error:", str(e))
        return {"status": "error", "message": str(e)}

# Directory for your saved reference chart patterns
REFERENCE_DIR = "reference_patterns"
os.makedirs(REFERENCE_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}


def compare_images(img1, img2):
    """Simple OpenCV structural similarity comparison"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize to same dimensions
    h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
    img1_gray = cv2.resize(img1_gray, (w, h))
    img2_gray = cv2.resize(img2_gray, (w, h))

    # Compute SSIM
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return float(score)

REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
print(f"📁 Reference directory: {REFERENCE_DIR}")
print("📁 Files found:", glob.glob(os.path.join(REFERENCE_DIR, "*")))

from fastapi import FastAPI, File, UploadFile
import os
import shutil
import os
import shutil
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

REFERENCE_DIR = "reference_patterns"
os.makedirs(REFERENCE_DIR, exist_ok=True)

@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a reference chart directly (manual upload)."""
    try:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        save_path = os.path.join(REFERENCE_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "path": save_path
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload_reference_json")
async def upload_reference_json(request: Request):
    """
    Accept Base44-style JSON uploads where the image is already hosted (file_url).
    Downloads the image to the reference_patterns folder.
    """
    try:
        data = await request.json()
        file_url = data.get("file_url")
        title = data.get("title", "unnamed")

        if not file_url:
            return {"status": "error", "message": "Missing 'file_url' in request"}

        os.makedirs(REFERENCE_DIR, exist_ok=True)
        save_path = os.path.join(REFERENCE_DIR, f"{title}.png")

        # Download the image and save it
        resp = requests.get(file_url, timeout=10)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)

        return {
            "status": "success",
            "message": f"File '{title}' saved to reference library.",
            "path": save_path
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

import cv2
import numpy as np
from fastapi import UploadFile, File

# --- Core reusable analysis logic ---
async def analyze_uploaded_chart(image_path: str):
    """Analyze an uploaded chart against stored reference patterns."""
    try:
        reference_files = [
            f for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not reference_files:
            return {"status": "no_references", "message": "No reference charts found."}

        # Load uploaded image
        uploaded_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if uploaded_img is None:
            return {"status": "error", "message": "Invalid uploaded image."}

        best_match = None
        best_score = -1

        # Compare with all references
        for ref_file in reference_files:
            ref_path = os.path.join(REFERENCE_DIR, ref_file)
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

            if ref_img is None:
                continue

            # Resize for uniformity
            ref_img_resized = cv2.resize(ref_img, (uploaded_img.shape[1], uploaded_img.shape[0]))

            # Compute similarity using correlation coefficient
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


# --- Normal API route version (still usable from frontend if needed) ---
@app.post("/analyze")
async def analyze_chart(file: UploadFile = File(...)):
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await analyze_uploaded_chart(temp_path)
    os.remove(temp_path)
    return result

    """Compare uploaded image to reference patterns."""
    reference_files = os.listdir(REFERENCE_DIR)
    if not reference_files:
        return {"status": "error", "message": "No reference images found. Please upload some first."}

    # You can replace this with your actual image comparison logic:
    result = {"comparison_count": len(reference_files), "matched": False}
    print(f"🧩 Comparing uploaded chart against {len(reference_files)} references...")
    return {"status": "success", "result": result}

@app.post("/analyze")
async def analyze_chart(file: UploadFile = File(...)):
    """Analyze uploaded chart and compare against stored reference patterns"""
    try:
        # Read uploaded image
        image_data = await file.read()
        uploaded_image = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))
        uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

        # Compare with each reference chart
        results = []
        for ref_path in glob.glob(os.path.join(REFERENCE_DIR, "*")):
            ref_img = cv2.imread(ref_path)
            similarity = compare_images(uploaded_image, ref_img)
            results.append({
                "reference": os.path.basename(ref_path),
                "similarity": round(similarity, 3)
            })

        # Pick top match
        if results:
            best_match = max(results, key=lambda x: x["similarity"])
            detected = best_match["reference"]
            confidence = best_match["similarity"]
        else:
            detected = None
            confidence = 0.0

        return {
            "status": "success",
            "detected_pattern": detected,
            "confidence": confidence,
            "matches": results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

from fastapi import UploadFile, File
import shutil
import os

@app.post("/process_chart")
async def process_chart(file: UploadFile = File(...)):
    """
    Upload a chart, save it to the reference folder, and automatically analyze it.
    """
    try:
        # Ensure directory exists
        os.makedirs(REFERENCE_DIR, exist_ok=True)

        # Save uploaded file
        save_path = os.path.join(REFERENCE_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Log that the upload succeeded
        print(f"✅ Saved new chart: {save_path}")

        # Automatically analyze the uploaded image
        from main import analyze_uploaded_chart  # Import your analyze logic if separated
        analysis_result = await analyze_uploaded_chart(save_path)

        return {
            "status": "success",
            "message": f"Chart '{file.filename}' uploaded and analyzed successfully.",
            "file_path": save_path,
            "analysis_result": analysis_result,
        }

    except Exception as e:
        print(f"❌ Error in /process_chart: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
