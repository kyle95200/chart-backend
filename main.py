from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import io
import os
import glob

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
    score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(score)
    return float(max_val)

REFERENCE_DIR = os.path.join(os.getcwd(), "reference_patterns")
print(f"📁 Reference directory: {REFERENCE_DIR}")
print("📁 Files found:", glob.glob(os.path.join(REFERENCE_DIR, "*")))

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
