# Real-Time CV Queue Management System

## Project layout

```
queue_system/
├── requirements.txt
├── tracker.py       # CentroidTracker — gives each person a stable ID across frames
├── vision.py         # QueueVision — YOLO inference + ROI logic + smoothing, runs in a thread
├── main.py            # FastAPI app — exposes /api/queue-status, serves the dashboard
├── static/
│   └── index.html      # Polling dashboard (vanilla JS)
└── demo_queue.mp4    # <- YOU add this (any mp4 with people standing/walking)
```

## 1. Environment setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

What each package is for:
- `ultralytics` — YOLOv8 model + inference wrapper (pulls in `torch` automatically)
- `opencv-python` — video I/O, ROI drawing, `pointPolygonTest`
- `fastapi` + `uvicorn[standard]` — the API server
- `numpy` — centroid/distance math in the tracker
- `python-multipart` — required by FastAPI's `StaticFiles`/form handling

The first time you run the app, Ultralytics will auto-download
`yolov8n.pt` (~6MB) into the working directory. No manual download step
needed.

## 2. Add a video file

Drop any `.mp4` with people in frame into the project root and name it
`demo_queue.mp4` (or change `VIDEO_SOURCE` in `main.py`). A phone recording
of a line at a coffee shop, a public webcam recording, or a stock "queue"
clip all work fine. To use a live webcam instead, set:

```python
VIDEO_SOURCE = 0
```

## 3. Calibrate the ROI polygon

The `ROI_POLYGON` list in `main.py` is a placeholder. Your video's actual
resolution and queue layout won't match it, so:

1. Run the server (see step 4).
2. Open `http://localhost:8000/api/video-feed` in a browser tab. This
   shows the raw feed with detections, IDs, and the current (probably
   wrong) polygon drawn in yellow.
3. Note where the real queue line is, in pixel coordinates of that video,
   and edit the four `(x, y)` tuples in `ROI_POLYGON` to trace it.
4. Restart the server to reload.

## 4. Run it

```bash
python main.py
```

Then open **http://localhost:8000** — that's the dashboard.
Endpoints:
- `GET /` — dashboard UI
- `GET /api/queue-status` — JSON: `{queue_count, raw_count, estimated_wait_minutes, tracked_people, timestamp}`
- `GET /api/video-feed` — debug MJPEG stream (ROI + boxes + IDs overlay)

## Architecture notes

**Why a background thread instead of running inference inside the request
handler?** If detection ran per-request, every dashboard poll would pay
the full YOLO inference cost and the frame rate would be tied to your
polling interval instead of the video's actual pace. Instead, `vision.py`
runs continuously on its own thread, constantly updating a small shared
state dict behind a lock. The FastAPI endpoint just reads that dict — it
responds in microseconds regardless of how expensive inference is.

**Why two layers of smoothing (tracker grace period + moving average)?**
They solve different problems:
- The **tracker's `max_disappeared`** (in `tracker.py`) solves *identity*
  loss — a specific person briefly vanishing from detection due to
  occlusion. Their ID and last known position are preserved for N frames,
  so they don't get dropped from the ROI count.
- The **rolling average buffer** (in `vision.py`) solves *count* jitter —
  even with stable IDs, minor per-frame noise (one frame a nearby
  non-queue person edges into the ROI boundary, etc.) can bounce the raw
  number around. Averaging the last 8 raw counts and rounding gives you a
  number that changes only when the trend actually changes.

**Why bottom-center instead of box-center for the ROI test?** A queue ROI
is a floor-space, not a body-space. The bottom edge of a person's bounding
box is the closest thing YOLO gives you to "where their feet are," which
is what actually determines whether they're standing in the queue area.
Box-center drifts up/down with how much of the person is visible in
frame, which would make a standing person incorrectly test in/out of the
ROI depending on box height alone.

## Tuning knobs (all in `main.py`'s `QueueVision(...)` call)

| Param | Effect |
|---|---|
| `minutes_per_person` | Multiplier for wait-time estimate |
| `smoothing_window` | Larger = smoother but slower to react to real changes |
| `max_disappeared` | Larger = more occlusion tolerance, but slower to drop people who actually left |
| `model_path` | Swap `yolov8n.pt` → `yolov8s.pt` for more accuracy at the cost of speed |
