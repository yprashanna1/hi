from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import cv2
from ultralytics import YOLO
from typing import Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")

model = YOLO(model_path)

@app.get("/")
def home():
    return {"message": "Accident detection online"}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    return_annotated_video: bool = False
):
    tmp_file = f"/tmp/{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    with open(tmp_file, "wb") as f:
        f.write(await file.read())

    ext = file.filename.lower().split('.')[-1]
    is_video = ext in ["mp4", "avi", "mov", "mkv", "wmv"]

    if not is_video:
        frame = cv2.imread(tmp_file)
        results = model.predict(frame, conf=0.25)
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x_center, y_center, w, h = box.xywh[0]
            boxes.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": conf,
                "xywh": [float(x_center), float(y_center), float(w), float(h)]
            })
        os.remove(tmp_file)
        return {"type": "image", "detections": boxes}

    cap = cv2.VideoCapture(tmp_file)
    if not cap.isOpened():
        os.remove(tmp_file)
        return JSONResponse({"error": "Cannot open video"}, status_code=400)

    out_path = f"/tmp/{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if return_annotated_video:
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frames_info = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25)
        if return_annotated_video:
            annotated = results[0].plot()
            writer.write(annotated)
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x_center, y_center, bw, bh = box.xywh[0]
            boxes.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": conf,
                "xywh": [float(x_center), float(y_center), float(bw), float(bh)]
            })
        frames_info.append({"frame": idx, "detections": boxes})
        idx += 1

    cap.release()
    os.remove(tmp_file)
    if return_annotated_video:
        writer.release()
        def video_stream():
            with open(out_path, "rb") as f:
                yield from f
            os.remove(out_path)
        return StreamingResponse(video_stream(), media_type="video/mp4")
    else:
        if os.path.exists(out_path):
            os.remove(out_path)
        return {"type": "video", "frames": len(frames_info), "results": frames_info}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)
