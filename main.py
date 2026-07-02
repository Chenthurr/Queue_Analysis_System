"""
main.py
-------
FastAPI server. Starts the QueueVision pipeline on a background thread at
startup, and exposes:

  GET /api/queue-status   -> current smoothed count + wait time (JSON)
  GET /api/video-feed     -> optional MJPEG debug stream (see the ROI/boxes
                              drawn live -- great for tuning ROI_POLYGON)
  GET /                    -> the dashboard (static/index.html)

The vision pipeline runs independently of request handling: FastAPI just
reads whatever QueueVision.get_state() currently holds. This decoupling is
what lets the endpoint respond instantly regardless of inference speed.
"""
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from vision import QueueVision

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Point this at your own .mp4, or use 0 for a live webcam.
VIDEO_SOURCE = "demo_queue.mp4"

# ROI polygon in (x, y) pixel coordinates of the SOURCE video's resolution.
# demo_queue.mp4 (intel-iot-devkit "people-detection.mp4") is 768x432 @ 12fps.
# Calibrated against that video's walking area -- re-tune via
# /api/video-feed if you swap in your own footage of a different resolution.
ROI_POLYGON = [
    (20, 130),
    (748, 130),
    (748, 420),
    (20, 420),
]

vision = QueueVision(
    video_source=VIDEO_SOURCE,
    roi_points=ROI_POLYGON,
    model_path="yolov8n.pt",       # nano model: best speed for real-time
    minutes_per_person=2.0,        # assumption per the spec
    smoothing_window=8,            # ~8 frames of moving-average smoothing
    max_disappeared=15,            # ~1s of occlusion tolerance at 15fps
)

app = FastAPI(title="Queue Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    vision.start()


@app.on_event("shutdown")
def shutdown_event():
    vision.stop()


@app.get("/api/queue-status")
def queue_status():
    """
    Returns e.g.:
    {
      "queue_count": 4,
      "raw_count": 4,
      "estimated_wait_minutes": 8.0,
      "tracked_people": 5,
      "timestamp": 1751234567.12
    }
    """
    return vision.get_state()


def _mjpeg_generator():
    while True:
        frame = vision.latest_frame
        if frame is None:
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/api/video-feed")
def video_feed():
    """Optional debug view: open this URL directly in a browser tab to see
    the ROI polygon, boxes, and IDs drawn live on the video."""
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
