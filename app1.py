from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

origins = [
    "https://caikey-17.github.io",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình YOLOv8
model = YOLO("yolov8n.pt")

allowed_labels = {"laptop", "cell phone", "keyboard", "mouse", "tv", "monitor", "computer"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image_np = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model(image)

    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            conf = float(box.conf[0])

            if label.lower() in allowed_labels:
                detected_objects.append({"label": label, "confidence": conf})

    return {"objects": detected_objects}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
